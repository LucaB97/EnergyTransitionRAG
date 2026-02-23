def determine_retry_reason(metrics, threshold=0.3):
    
    failures = {
        # High when one paper dominates despite low overall coverage
        "source_diversity": (
            metrics["paper_dominance"] - (1 - metrics["paper_coverage"])
        ),

        # High when most sentences rely on a single source
        "corroboration": 1 - metrics["multi_source_sentence_ratio"],

        # High when many retrieved chunks were ignored
        "evidence_utilization": 1 - metrics["chunk_coverage"],
    }

    # Find worst failure mode
    retry_reason, severity = max(
        failures.items(),
        key=lambda x: x[1]
    )

    # If even the worst failure is mild → no retry
    if severity < threshold:
        return None

    return retry_reason
