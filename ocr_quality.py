"""Bounded, offline text-region refinement; boxes remain in full-frame coordinates."""
import re
import unicodedata


def clean_ocr_text(text):
    text = unicodedata.normalize("NFC", str(text or ""))
    text = "".join(c for c in text if unicodedata.category(c) != "Cf")
    return re.sub(r"\s+", " ", text).strip()


def overlap(a, b):
    x = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    y = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = x * y
    return area / max(1e-9, (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - area)


def merge_readings(base, additions):
    merged = [dict(t, text=clean_ocr_text(t["text"])) for t in base]
    for item in additions:
        item = dict(item, text=clean_ocr_text(item["text"]))
        if not item["text"]:
            continue
        duplicate = None
        contained_fragment = False
        remove = []
        for i, old in enumerate(merged):
            a, b = old["box"], item["box"]
            intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
            old_area = max(1e-9, (a[2]-a[0])*(a[3]-a[1]))
            item_area = max(1e-9, (b[2]-b[0])*(b[3]-b[1]))
            same_text = item["text"].casefold() == old["text"].casefold()
            if intersection/item_area >= .75 and (
                    (item_area < old_area*.9 and len(item["text"]) <= len(old["text"])) or same_text):
                # A tile can see only a word from an already recognized line.
                contained_fragment = True
                break
            if intersection/old_area >= .75 and old_area < item_area*.9 and len(old["text"]) <= len(item["text"]):
                remove.append(i)
            elif overlap(a, b) >= .5:
                duplicate = i
        if contained_fragment:
            continue
        if duplicate is not None:
            if item["conf"] > merged[duplicate]["conf"]:
                merged[duplicate] = item
        else:
            merged.append(item)
        for i in reversed(remove):
            merged.pop(i)
    return sorted(merged, key=lambda t: (round(t["box"][1], 2), t["box"][0]))


def refine_regions(image, texts, read, *, max_regions=8):
    """Retry small/uncertain boxes; tiled contrast passes also recover missed text.

    `read` accepts an image and returns normalized boxes. Failures propagate so
    incomplete recognition is never marked as the current successful version.
    """
    import cv2
    h, w = image.shape[:2]
    boxes = []
    for item in sorted(texts, key=lambda t: (t["conf"], t["box"][1])):
        x1, y1, x2, y2 = item["box"]
        if item["conf"] < .8 or (y2-y1)*h < 40:
            boxes.append((max(0, int(x1*w)-8), max(0, int(y1*h)-8),
                          min(w, int(x2*w)+8), min(h, int(y2*h)+8)))
    # Reserve up to four overlapping tiles for text the first detector missed.
    tiles = [(int(x*w), int(y*h), int(min(1, x+.6)*w), int(min(1, y+.6)*h))
             for y in (0, .4) for x in (0, .4)]
    regions = (boxes[:max(0, max_regions-4)] + tiles)[:max_regions]
    additions = []
    for x1, y1, x2, y2 in regions:
        if x2 <= x1 or y2 <= y1:
            continue
        crop = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        scale = min(2.0, 1600 / max(gray.shape))
        gray = cv2.resize(gray, (max(1, round(gray.shape[1]*scale)),
                                max(1, round(gray.shape[0]*scale))), interpolation=cv2.INTER_CUBIC)
        for item in read(gray):
            bx1, by1, bx2, by2 = item["box"]
            box = [(x1+bx1*(x2-x1))/w, (y1+by1*(y2-y1))/h,
                   (x1+bx2*(x2-x1))/w, (y1+by2*(y2-y1))/h]
            additions.append(dict(item, box=[round(min(1, max(0, b)), 4) for b in box]))
    return merge_readings(texts, additions)
