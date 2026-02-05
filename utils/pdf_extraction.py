# import statistics

# def detect_layout(page):
#     words = page.extract_words(use_text_flow=True)
#     if not words:
#         return "one_column"

#     page_width = page.width

#     lines = {}
#     for w in words:
#         key = round(w["top"], 1)
#         lines.setdefault(key, []).append(w)

#     line_widths = []
#     for line_words in lines.values():
#         x0 = min(w["x0"] for w in line_words)
#         x1 = max(w["x1"] for w in line_words)
#         line_widths.append((x1 - x0) / page_width)

#     if not line_widths:
#         return "one_column"

#     median_width = statistics.median(line_widths)

#     # Conservative threshold
#     if median_width < 0.55:
#         return "two_column"

#     return "one_column"



def extract_text_two_columns(page):
    """
    Extract text from a PDF page assuming a two-column layout.

    The page is split vertically into left and right halves, which are
    extracted separately and then concatenated in reading order.

    Parameters
    ----------
    page : pdfplumber.page.Page
        A pdfplumber Page object.

    Returns
    -------
    str
        Extracted text with left column followed by right column.
    """   
     
    width = page.width
    midpoint = width / 2

    left_bbox = (0, 0, midpoint, page.height)
    right_bbox = (midpoint, 0, width, page.height)

    left = page.crop(left_bbox).extract_text(layout=True) or ""
    right = page.crop(right_bbox).extract_text(layout=True) or ""

    return left + "\n\n" + right