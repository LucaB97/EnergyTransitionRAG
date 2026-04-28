import pandas as pd
import time
import logging
import json
from collections import defaultdict

from utils.prompt import SCOPE_CLASSIFIER_PROMPT, RELEVANCE_JUDGE_PROMPT, QUERY_EXPANDER_PROMPT, TASK_HEADER, CORE_SYNTHESIS_INSTRUCTIONS, RETRY_PROMPTS
from utils.citations import build_citation_index, build_sources, CitationStyle, FORMATTERS, resolve_answer_citations, remove_citations_inside_text 
from utils.chunking import deduplicate

from .evaluation.evidence_analysis import aggregate_evidence, extract_sentence_paper_ids, compute_grounding_metrics
from .evaluation.confidence import evaluate_grounding_quality, evaluate_confidence_profile
from .evaluation.retry_policy import reason_retry_grounding
from .postprocessing.response_builder import build_query_response

from schemas.request import QueryRequest
from schemas.response import Sentence

class RAGPipeline:

    def __init__(
        self,
        metadata,
        scope_classifier,
        normalizer,
        retriever,
        relevance_profiler,
        evidence_relevance_judge,
        query_expander,
        synthesizer
    ):
        self.metadata = metadata
        self.scope_classifier = scope_classifier
        self.normalizer = normalizer
        self.retriever = retriever
        self.relevance_profiler = relevance_profiler
        self.evidence_relevance_judge = evidence_relevance_judge
        self.query_expander = query_expander
        self.synthesizer = synthesizer
        self.topN = 15


    def initialize_output_meta(self):

        return {
            "total_time_sec": None,
            "query_classification": {
                "in_scope": True,
                "strategy": {
                    "llm_type": type(self.scope_classifier.llm).__name__,
                    "model": self.scope_classifier.llm.model_name,
                    "temperature": self.scope_classifier.llm.temperature,
                },
            },
            "retrieval": {},
            "profiling": {},
            "evidence_relevance": {},
            "errors": {},
            "synthesis": {}
        }
    

    def run(self, request: QueryRequest):
        logger = logging.getLogger(__name__)
        start_time = time.perf_counter()
        
        meta = self.initialize_output_meta()
        pipeline_status = "success"

        user_query = request.question
        
        #
        # --- Zero-shot classification of scope---
        #
        in_scope = self.scope_classifier.is_in_scope(user_query, SCOPE_CLASSIFIER_PROMPT)
        
        if not in_scope:
            total_time = time.perf_counter() - start_time
            pipeline_status = "out_of_scope"
            limitations = ["This query is outside the scope of the system"]
            meta["total_time_sec"] = total_time
            meta["query_classification"]["in_scope"] = False
            confidence_profile = evaluate_confidence_profile(pipeline_status, reason="Out of scope")
            return build_query_response(user_query, pipeline_status, limitations, meta=meta, confidence=confidence_profile)

        #
        # --- Retrieval ---
        #
        norm_query = self.normalizer.normalize(user_query)
        topk_faiss, topk_bm25 = request.topk_faiss, request.topk_bm25
        query_expansion = False
        relevant_count_exp = None
        queries = [user_query]

        t0 = time.perf_counter()
        retrieved_chunks = self.retriever.search(user_query, norm_query, topk_faiss=topk_faiss, topk_bm25=topk_bm25)
        retrieval_time = time.perf_counter() - t0

        meta["retrieval"] = {
            "candidate_pool_size": len(retrieved_chunks),
            "retrieval_time_sec": round(retrieval_time, 3),
            "retriever_info": {
                "strategy": "hybrid",
                "semantic": {
                    "embedding_backend": type(self.retriever.semantic_retriever.embedding_fn).__name__,
                    "embedding_model": getattr(self.retriever.semantic_retriever.embedding_fn, "model_name", None),
                    "faiss_index_type": type(self.retriever.semantic_retriever.index).__name__,
                    "chunks_requested": topk_faiss,
                },
                "bm25": {
                    "enabled": hasattr(self.retriever, "bm25_retriever"),
                    "chunks_requested": topk_bm25,
                }
            },
            "query_expansion": {
                "state": False,
                "num_queries": 1,
                "strategy": {},
            },  
        }
        
        if not retrieved_chunks:
            pipeline_status = "retrieval_failed"
            limitations = ["No documents could be retrieved for this question"]
            confidence_profile = evaluate_confidence_profile(pipeline_status, reason="Empty retrieval")
            trace = {
                "queries": queries
            }
            return build_query_response(user_query, pipeline_status, limitations, meta=meta, confidence=confidence_profile)

        #
        # --- Reranking & Relevance Evaluation ---
        #
        t1 = time.perf_counter()
        reranked_chunks = self.relevance_profiler.rerank(user_query, retrieved_chunks)
        profiling_time = time.perf_counter() - t1

        meta["profiling"] = {
            "class": self.relevance_profiler.reranker,
            "model": self.relevance_profiler.reranker.model_name,
            "profiling_time_sec": round(profiling_time, 3),
        }

        top_chunks = reranked_chunks[:self.topN]
        texts = [chunk["text"] for chunk in top_chunks]
        
        try:
            t2 = time.perf_counter()
            relevance_scores = self.evidence_relevance_judge.judge_relevance_unified(user_query, texts, RELEVANCE_JUDGE_PROMPT)
            relevant_count = sum(relevance_scores)
            relevance_evaluation_time = time.perf_counter() - t2
            
            meta["evidence_relevance"] = {
                "evaluation_time_sec": round(relevance_evaluation_time, 3),
                "evaluation_info": {
                    "llm_type": type(self.evidence_relevance_judge.llm).__name__,
                    "model": self.evidence_relevance_judge.llm.model_name,
                    "temperature": self.evidence_relevance_judge.llm.temperature,
                },
            }

        except ValueError as e:
            pipeline_status = "relevance_evaluation_error"
            limitations=["An error occurred during the evaluation of the evidence. Please try again."]
            
            meta["errors"] = {	
                "evidence_evaluation_error": {
                    "state": True,
                    "total_attempts": self.synthesizer.max_attempts,
                    "last_error": str(e)
                }
            }

            confidence_profile = evaluate_confidence_profile(pipeline_status, reason="Evidence evaluation error")
            trace = {
                "queries": queries
            }
            return build_query_response(user_query, pipeline_status, limitations, meta=meta, confidence=confidence_profile) 


        #
        # --- Retrieval retry ---
        #

        if relevant_count < 8:
            
            if not query_expansion:

                query_expansion = True
                expanded_queries = self.query_expander.produce_expansion(user_query, QUERY_EXPANDER_PROMPT)
                
                if isinstance(expanded_queries, str):
                    expanded_queries = json.loads(expanded_queries)
                
                if isinstance(expanded_queries, list):
                    queries = [user_query] + expanded_queries[:3]            
                
                retrieved_chunks = []

                t0 = time.perf_counter()
                for q in queries:
                    if self.normalizer:
                        norm_q = self.normalizer.normalize(q)
                    else:
                        norm_q = None
                    retrieved_chunks.extend(self.retriever.search(q, norm_q, topk_faiss=topk_faiss, topk_bm25=topk_bm25))
                retrieval_time += time.perf_counter() - t0
                
                retrieved_chunks = deduplicate(retrieved_chunks)

                meta["retrieval"]["candidate_pool_size"] = len(retrieved_chunks)
                meta["retrieval"]["retrieval_time_sec"] = round(retrieval_time, 3)

                meta["retrieval"]["query_expansion"]["state"] = query_expansion
                meta["retrieval"]["query_expansion"]["num_queries"] = len(queries)
                meta["retrieval"]["query_expansion"]["strategy"] = {
                    "llm_type": type(self.query_expander.llm).__name__,
                    "model": self.query_expander.llm.model_name, 
                    "temperature": self.query_expander.llm.temperature,
                }

                if not retrieved_chunks:
                    pipeline_status = "retrieval_failed"
                    limitations = ["No documents could be retrieved for this question"]
                    confidence_profile = evaluate_confidence_profile(pipeline_status, reason="Empty retrieval")
                    trace={"queries": queries}
                    return build_query_response(user_query, pipeline_status, limitations, meta=meta, confidence=confidence_profile, trace=trace)
                
                t1 = time.perf_counter()
                reranked_chunks_exp = self.relevance_profiler.rerank(user_query, retrieved_chunks)
                profiling_time += time.perf_counter() - t1
                meta["profiling"]["profiling_time_sec"] = round(profiling_time, 3)

                top_chunks_exp = reranked_chunks_exp[:self.topN]
                texts = [chunk["text"] for chunk in top_chunks_exp]
                
                try:
                    t2 = time.perf_counter()
                    relevance_scores_exp = self.evidence_relevance_judge.judge_relevance_unified(user_query, texts, RELEVANCE_JUDGE_PROMPT)
                    relevant_count_exp = sum(relevance_scores_exp)
                    relevance_evaluation_time += time.perf_counter() - t2
                    meta["evidence_relevance"]["evaluation_time_sec"] = round(relevance_evaluation_time, 3)
                except ValueError as e:
                    pipeline_status = "relevance_evaluation_error"
                    limitations=["An error occurred during the evaluation of the evidence. Please try again."]
                    meta["errors"] = {	
                        "evidence_evaluation_error": {
                            "state": True,
                            "total_attempts": self.synthesizer.max_attempts,
                            "last_error": str(e)
                        }
                    }
                    confidence_profile = evaluate_confidence_profile(pipeline_status, reason="Evidence evaluation error")
                    return build_query_response(user_query, pipeline_status, limitations, meta=meta, confidence=confidence_profile) 
 

        #
        # --- Synthesis ---
        #    
        meta["synthesis"] = {
            "chunks_selected": self.topN,
            "synthesis_time_sec": None,
            "synthesizer_info": {
                "llm_type": type(self.synthesizer.llm).__name__,
                "model": self.synthesizer.llm.model_name,
                "max_tokens": self.synthesizer.llm.max_tokens,
                "temperature": self.synthesizer.llm.temperature,
            },
            "synthesis_retry": {
                "attempted": None,
                "total_attempts": None,
                "retry_triggers": None
            },
        }
        
        if relevant_count_exp:
            if relevant_count_exp > relevant_count:
                relevant_count = relevant_count_exp
                top_chunks = top_chunks_exp
            
        for c in top_chunks:
            c_metadata = self.metadata.loc[c["paper_id"]]
            c['title'] = str(c_metadata['title'])
            c['authors'] = str(c_metadata['authors'])
            c['year'] = int(c_metadata['year'])
            c['journal'] = str(c_metadata['journal'])
            c['first_tag'] = (None if pd.isna(c_metadata['first_tag']) else str(c_metadata['first_tag']))
            c['second_tag'] = (None if pd.isna(c_metadata['second_tag']) else str(c_metadata['second_tag']))

        source_lookup = {
            c["chunk_id"]: c
            for c in top_chunks
        }

        max_attempts = 2
        attempt = 0
        synthesis_output = None
        best_output, best_score = None, -1
        last_error = None
        retry_triggers = []

        prompt = TASK_HEADER + CORE_SYNTHESIS_INSTRUCTIONS

        t3 = time.perf_counter()

        while attempt < max_attempts:
            attempt += 1
            
            try:
                synthesis_output = self.synthesizer.synthesize(user_query, top_chunks, prompt)
            except ValueError as e:
                last_error = e
                logger.error("Synthesis failed after retries", exc_info=e)
                break  # hard failure → exit loop


            answer = synthesis_output["answer"]

            if not answer:
                synthesis_time = time.perf_counter() - t3
                total_time = time.perf_counter() - start_time

                limitations = synthesis_output.get("limitations") or ["No meaningful answer could be produced from the available literature"]

                meta["synthesis"]["synthesis_time_sec"] = round(synthesis_time, 3)
                meta["total_time_sec"] = round(total_time, 3)
                
                aggregation = aggregate_evidence(top_chunks)
                confidence_profile = evaluate_confidence_profile(pipeline_status, relevant_count, len(top_chunks), reason="Abstention")
                
                trace={
                "queries": queries,
                "chunks_provided_to_synthesizer": aggregation["chunks"],
                }
                
                return build_query_response(user_query, pipeline_status, limitations, meta=meta, confidence=confidence_profile, trace=trace)   

            ## Evidence metrics
            sentence_papers = extract_sentence_paper_ids(synthesis_output["answer"], source_lookup)

            used_chunks_ids = {
                cid
                for sentence in synthesis_output["answer"]
                for cid in sentence.get("citations", [])
            }

            aggregation = aggregate_evidence(top_chunks, used_chunks_ids)
            grounding_metrics = compute_grounding_metrics(aggregation, sentence_papers)

            grounding_score, grounding_flags = evaluate_grounding_quality(grounding_metrics)
            
            confidence_profile = evaluate_confidence_profile(pipeline_status, relevant_count, len(top_chunks), grounding_score, grounding_metrics, grounding_flags)
            
            if grounding_score >= best_score:
                best_output = synthesis_output
                best_sentence_papers = sentence_papers
                best_aggregation = aggregation
                best_grounding_metrics = grounding_metrics
                best_confidence = confidence_profile
                best_score = grounding_score

            # --- Retry decision ---
            if grounding_score < 0.5 and attempt < max_attempts:
                retry_reason = reason_retry_grounding(grounding_metrics)
            else:
                retry_reason = None             

            if grounding_metrics and retry_reason and attempt < max_attempts:
                retry_triggers.append(retry_reason)
                logger.info(
                    "Retrying synthesis due to weak grounding",
                    extra={"grounding_metrics": grounding_metrics}
                )
                prompt = TASK_HEADER + CORE_SYNTHESIS_INSTRUCTIONS + RETRY_PROMPTS[retry_reason]
                continue
            else:
                break  # synthesis accepted

        synthesis_time = time.perf_counter() - t3
        total_time = time.perf_counter() - start_time
        meta["synthesis"]["synthesis_time_sec"] = round(synthesis_time, 3)
        meta["total_time_sec"] = round(total_time, 3)
        
        # --- Failure fallback ---
        if last_error and not best_output:
            pipeline_status = "generation_error"
            limitations=["The system was unable to generate a reliable answer this time. Please try again."]
            
            meta["errors"] = {	
                "generation_error": {
                    "state": True,
                    "total_attempts": self.synthesizer.max_attempts,
                    "last_error": str(last_error)
                },
            }

            confidence_profile = evaluate_confidence_profile(pipeline_status, reason="Generation error")
            return build_query_response(user_query, pipeline_status, limitations, meta=meta, confidence=confidence_profile)  

        #
        # --- Output preparation ---
        #
        if attempt>1:
            meta["synthesis"]["synthesis_retry"]["attempted"] = attempt>1
            meta["synthesis"]["synthesis_retry"]["total_attempts"] = attempt
            meta["synthesis"]["synthesis_retry"]["retry_triggers"] = retry_triggers

        citation_index = build_citation_index(best_sentence_papers)
        sources = build_sources(citation_index, source_lookup)

        style = CitationStyle.NUMERIC
        resolved_answer = resolve_answer_citations(best_output["answer"], source_lookup, citation_index, FORMATTERS[style])
        resolved_answer = remove_citations_inside_text(resolved_answer) ## remove references from synthesis text

        trace={
            "queries": queries,
            "grounding_metrics": best_grounding_metrics,
            "chunks_provided_to_synthesizer": best_aggregation["chunks"],
            "paper_stats": [
                {"paper_id": pid, **stats}
                for pid, stats in best_aggregation["paper_stats"].items()
            ]
        }

        return build_query_response(user_query, pipeline_status, limitations=best_output["limitations"], answer=[Sentence(**s) for s in resolved_answer], 
                            sources=sources, meta=meta, confidence=best_confidence, trace=trace)
