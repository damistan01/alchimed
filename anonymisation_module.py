"""
Anonymisation Module for Medical Blood Test Images.

Uses DocTR OCR to detect text regions with bounding boxes, then classifies
each region as sensitive (PII) or safe (medical data) using:
  1. Regex patterns (CNP, phone numbers, etc.)
  2. Keyword proximity (label fields like "Nume:", "Adresa:", "Tel:", etc.)
  3. Structural heuristics (header zone exclusion, barcode/QR detection)

Only the sensitive regions are covered with solid rectangles, preserving
all medical test data, reference ranges, and units.
"""

import os
import re
import cv2
import numpy as np
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# ---------------------------------------------------------------------------
# Shared DocTR model — import from vision_module if already loaded, otherwise
# initialise here so the module can also run standalone.
# ---------------------------------------------------------------------------
try:
    from vision_module import _doctr_model
    print("[Anon] Reusing DocTR model from vision_module.")
except ImportError:
    print("[Anon] Loading DocTR OCR model (standalone)...")
    _doctr_model = ocr_predictor(
        det_arch='db_resnet50',
        reco_arch='crnn_vgg16_bn',
        pretrained=True,
        assume_straight_pages=True
    )
    print("[Anon] DocTR ready (standalone).")


# ========================== SENSITIVITY RULES ==============================

# --- 1. PII label keywords --------------------------------------------------
#     Single-word labels that directly identify PII fields.
PII_SINGLE_WORD_LABELS = [
    "nume", "prenume",
    "cnp", "c.n.p", "c.n.p.",
    "adresa",
    "telefon", "tel", "tel/fax", "telfax",
    "mobil",
    "medic", "doctor",
    # English equivalents
    "name", "address", "phone", "mobile", "physician",
    "ssn",
]

#     Multi-word labels: the FIRST word triggers a look-ahead to form the full
#     label.  E.g. "Data" + "nasterii:" -> "data nasterii".
PII_MULTI_WORD_LABELS = [
    "nume pacient", "prenume pacient",
    "data nasterii",
    "cod pacient", "cod de bare",
    "numar cerere",
    # English equivalents
    "patient name", "first name", "last name",
    "social security", "patient id", "patient code",
]

# Combined list for backwards-compatibility in helper checks
PII_LABEL_KEYWORDS = PII_SINGLE_WORD_LABELS + PII_MULTI_WORD_LABELS

# First words of multi-word labels (used to trigger look-ahead)
_MULTI_LABEL_FIRST_WORDS = set()
for _lbl in PII_MULTI_WORD_LABELS:
    _MULTI_LABEL_FIRST_WORDS.add(_lbl.split()[0])

# Labels where values may appear on the line BELOW (not just to the right)
BELOW_VALUE_LABELS = {"telefon", "mobil", "adresa", "cod pacient", "cod de bare"}

# --- 2. Regex patterns that are *intrinsically* PII regardless of context ---
PII_REGEX_PATTERNS = [
    # Romanian CNP: exactly 13 digits (possibly with spaces/dots)
    re.compile(r'\b[1-9]\d{2}\s?\d{4}\s?\d{6}\b'),
    # Romanian phone: 07xx/02xx style, 10 digits
    re.compile(r'\b0[237]\d{8}\b'),
]

# --- 3. Words/tokens that are definitely SAFE (medical terms, units, etc.) ---
SAFE_KEYWORDS = [
    # Section headers
    "hematologie", "biochimie", "coagulare", "hematology", "biochemistry",
    "hemoleucograma", "hemoleucograma completa", "rezultat",
    # Common test names
    "hemoglobina", "hematocrit", "eritrocite", "leucocite", "trombocite",
    "neutrofile", "limfocite", "monocite", "eozinofile", "bazofile",
    "vem", "hem", "chem", "rdw", "mcv", "mch", "mchc", "mpv",
    "colesterol", "hdl", "ldl", "trigliceride", "creatinina",
    "uree", "glicemie", "alt", "ast", "gpt", "got",
    "proteina", "crp", "vsh", "inr", "fibrinogen",
    "hematii", "rbc", "wbc", "hgb", "hct", "plt",
    # Units
    "g/dl", "mg/dl", "u/l", "mmol/l", "fl", "pg",
    "*10^3/ul", "*10^6/ul", "10^3/ul", "10^6/ul",
    "mm/h", "mg/l", "iu/l",
    # Table headers
    "test", "rezultat", "rezultate", "um", "interval", "referinta",
    "analize", "denumire", "valori",
    # Page boilerplate
    "pagina", "data tiparire",
    # Common institution words we never want to redact
    "laborator", "medicale", "medical", "sucursala", "buletin",
    "acreditat", "certificat", "renar", "acreditare",
    "punct recoltare", "cabinet", "spital",
]

# --- 4. Safe label keywords — values after these are NOT PII ----------------
SAFE_LABEL_KEYWORDS = [
    "varsta", "sex", "vârsta",
    "data recoltare", "data recoltarii", "data receptie", "data validare",
    "data rezultat", "data inregistrarii",
    "data cerere", "data tiparire",
    "pagina",
    "interval de referinta", "interval biologic",
    "buletin de analize", "buletin analize",
    "raport", "rezultat partial",
    "punct recoltare", "cabinet", "spital",
    "recoltat",
]

# --- 5. Header zone threshold -----------------------------------------------
#     Text above this normalised Y coordinate is the clinic/institution header.
#     We skip certain PII label detection in this zone since it's the clinic's
#     own info, not the patient's. (Typically top ~18% of the page.)
CLINIC_HEADER_Y_THRESHOLD = 0.18

# Labels to ignore when they appear in the clinic header zone
HEADER_EXCLUDED_LABELS = {"tel", "telefon", "tel/fax", "telfax", "adresa",
                          "phone", "address"}

# Maximum horizontal gap (normalised) between a label's right edge and a
# candidate value's left edge. If the gap exceeds this, the candidate is
# likely in a different form column and should NOT be treated as the label's
# value. Typical form column gap is ~0.05; cross-column gap is ~0.25+.
MAX_LABEL_VALUE_GAP = 0.20


# ========================== CORE ALGORITHM =================================

def _normalise(text: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    return re.sub(r'\s+', ' ', text.strip().lower())


def _is_safe_by_content(text: str) -> bool:
    """Return True if the text is clearly medical/safe content."""
    norm = _normalise(text)

    # Pure numeric values (test results) — mostly safe, BUT be careful not to
    # classify phone numbers, IDs, CNPs, and birth dates as safe test results.
    # We consider a number "safe test result" if it's:
    # 1. A small number (e.g., 40.9, <1.2, 132) typical of lab values
    # 2. Contains medical math symbols (<, >, =, -)
    # Require decimal numbers to have digits after the dot/comma (prevents "73," from matching)
    if re.fullmatch(r'[<>=\-]?\s*\d+([.,]\d+)?\s*', norm):
        # Check if it's too long to be a normal test result (e.g., ID or phone)
        # Most medical results are <= 5 digits (e.g. 15000, 4.52)
        digits_only = re.sub(r'\D', '', norm)
        if len(digits_only) <= 5:
            return True
            
    # Also safe: specific unit patterns or reference ranges like "40-60" or "10^3/uL"
    if re.search(r'\d+\s*(mg/dl|u/l|%|/ul|g/dl|\*10\^|pg|fl|g/l|ui/ml|ng/ml|nmol/l|pmol/l)', norm):
        return True
    
    # Known safe keywords
    for kw in SAFE_KEYWORDS:
        if kw in norm:
            return True

    return False


def _is_pii_by_regex(text: str) -> bool:
    """Return True if the text matches an intrinsic PII pattern."""
    for pat in PII_REGEX_PATTERNS:
        if pat.search(text):
            return True
    return False


def _extract_pii_label(text: str) -> str | None:
    """
    If this SINGLE word looks like a PII label (possibly with value glued on),
    return the matching label keyword, otherwise None.

    Handles cases like:
      "Nume:"        -> "nume"
      "Medic:Dr."    -> "medic"  (also flags the glued value)
    """
    norm = _normalise(text)
    # Strip trailing colon and everything after for matching
    label_part = re.split(r'[:]', norm, maxsplit=1)[0].strip()
    # Remove trailing dots
    label_part = label_part.rstrip('.')

    for kw in PII_SINGLE_WORD_LABELS:
        if label_part == kw:
            return kw
    return None


def _extract_multiword_label(word_entries, start_idx: int) -> tuple[str | None, int]:
    """
    Check if the word at start_idx begins a multi-word PII label.

    Returns (label_keyword, word_count) if found, or (None, 0).
    E.g. for "Data" + "nasterii:" -> ("data nasterii", 2)
    """
    first_norm = _normalise(word_entries[start_idx]['text'])
    first_clean = re.split(r'[:]', first_norm, maxsplit=1)[0].strip().rstrip('.')

    if first_clean not in _MULTI_LABEL_FIRST_WORDS:
        return None, 0

    # Try combining 2-3 consecutive words to form a multi-word label
    for length in (3, 2):
        if start_idx + length > len(word_entries):
            continue
        combined_parts = []
        for offset in range(length):
            w = _normalise(word_entries[start_idx + offset]['text'])
            w = re.split(r'[:]', w, maxsplit=1)[0].strip().rstrip('.')
            combined_parts.append(w)
        combined = ' '.join(combined_parts)
        for kw in PII_MULTI_WORD_LABELS:
            if combined == kw:
                return kw, length

    return None, 0


def _has_glued_value(text: str) -> bool:
    """True if the word contains a label:value pattern (e.g. 'Medic:Dr.')."""
    return ':' in text and len(text.split(':', 1)) == 2 and len(text.split(':', 1)[1].strip()) > 0


def _is_safe_label(text: str) -> bool:
    """Return True if this text region is a safe-value label keyword."""
    norm = _normalise(text)
    norm = re.sub(r'[:.\s]+$', '', norm)
    for kw in SAFE_LABEL_KEYWORDS:
        if kw in norm:
            return True
    return False


def _boxes_on_same_row(box_a, box_b, tolerance_ratio=0.015):
    """Check if two bounding boxes share roughly the same vertical band."""
    mid_a = (box_a[0][1] + box_a[1][1]) / 2
    mid_b = (box_b[0][1] + box_b[1][1]) / 2
    return abs(mid_a - mid_b) < tolerance_ratio


def _box_is_right_of(box_a, box_b):
    """True if box_b's left edge is at or past box_a's right edge."""
    return box_b[0][0] > box_a[1][0] - 0.02


def _box_y_center(box):
    """Return the vertical center of a box (normalised)."""
    return (box[0][1] + box[1][1]) / 2


def _in_clinic_header(box):
    """True if the box is in the clinic header zone."""
    return _box_y_center(box) < CLINIC_HEADER_Y_THRESHOLD


def _detect_qr_barcode_regions(image_bgr):
    """
    Detect likely QR code / barcode regions using contour analysis.
    Returns a list of (x, y, w, h) rectangles in pixel coordinates.

    Tuned to detect dense square-ish regions (QR codes) and dense wide regions (1D barcodes).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h_img, w_img = image_bgr.shape[:2]

    # Binary threshold to find very dark regions
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # Use a wide kernel to merge 1D barcode lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    min_side = min(h_img, w_img) * 0.03

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Must not be too large (not the whole page)
        if w > w_img * 0.8 or h > h_img * 0.8:
            continue
            
        if w < min_side or h < min_side:
            continue

        aspect = w / max(h, 1)

        # Check fill density
        roi = binary[y:y+h, x:x+w]
        fill_ratio = np.count_nonzero(roi) / max(w * h, 1)

        # QR codes are square-ish
        is_qr = (0.7 < aspect < 1.4) and fill_ratio > 0.35
        
        # Barcodes are wide
        is_barcode = (2.0 < aspect < 15.0) and fill_ratio > 0.35

        if is_qr or is_barcode:
            regions.append((x, y, w, h))

    return regions


# ========================== MAIN FUNCTION ==================================

def anonymise_image(image_path: str, output_path: str | None = None,
                    redact_color=(0, 0, 0), margin_px=4,
                    debug=False) -> str:
    """
    Anonymise a medical blood test image by redacting PII regions.

    Parameters
    ----------
    image_path : str
        Path to the input image.
    output_path : str or None
        Where to save the anonymised image. If None, saves next to the
        original with an '_anon' suffix.
    redact_color : tuple
        BGR colour for the redaction rectangles (default: black).
    margin_px : int
        Extra pixels to extend around each redacted box.
    debug : bool
        If True, draw green boxes around safe words for visual debugging.

    Returns
    -------
    str : path to the saved anonymised image.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"[Anon] Processing: {os.path.basename(image_path)}")

    # --- Load image ----------------------------------------------------------
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    h_img, w_img = image_bgr.shape[:2]

    # --- Run DocTR OCR -------------------------------------------------------
    doc = DocumentFile.from_images(image_path)
    result = _doctr_model(doc)

    # Flatten all words with their bounding boxes (normalised 0..1)
    word_entries = []
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    word_entries.append({
                        'text': word.value,
                        'box': word.geometry,   # ((x0,y0),(x1,y1)) normalised
                        'redact': False,
                        'reason': '',
                    })

    print(f"[Anon] Found {len(word_entries)} word regions.")

    # --- Phase 1: Mark intrinsically PII words (regex) -----------------------
    for entry in word_entries:
        if _is_pii_by_regex(entry['text']):
            entry['redact'] = True
            entry['reason'] = 'regex_match'
            print(f"  [Phase1] Regex PII: '{entry['text']}'")

    # --- Phase 2: Label-based proximity scanning (STRICT same-row only) ------
    #     Handles both single-word labels ("Nume:") and multi-word labels
    #     ("Data nasterii:", "Cod pacient:", "Numar cerere:").
    processed_indices = set()  # skip words already consumed as part of a label

    for i, entry in enumerate(word_entries):
        if i in processed_indices:
            continue

        # Try multi-word label first (higher priority)
        label_kw, label_word_count = _extract_multiword_label(word_entries, i)
        if label_kw is not None:
            # Mark the label words as processed
            for offset in range(label_word_count):
                processed_indices.add(i + offset)
            # The last word of the label is the anchor for value scanning
            label_anchor = word_entries[i + label_word_count - 1]
            scan_start = i + label_word_count
        else:
            # Try single-word label
            label_kw = _extract_pii_label(entry['text'])
            if label_kw is None:
                continue
            label_anchor = entry
            scan_start = i + 1

        # --- Skip labels in the clinic header zone ---
        if label_kw in HEADER_EXCLUDED_LABELS and _in_clinic_header(entry['box']):
            print(f"  [Phase2] Skipping '{entry['text']}' — clinic header zone")
            continue

        # For "adresa", also check if it's a lab/collection-point address
        if label_kw == "adresa" and not _in_clinic_header(entry['box']):
            is_lab_address = False
            lab_context_words = ("punct", "recoltare", "cabinet", "spital",
                                 "laborator", "sucursala")
            search_start = max(0, i - 20)
            for k in range(search_start, i):
                prev_norm = _normalise(word_entries[k]['text'])
                if any(ctx in prev_norm for ctx in lab_context_words):
                    is_lab_address = True
                    break
            if is_lab_address:
                print(f"  [Phase2] Skipping '{entry['text']}' — lab/collection address")
                continue

        # If the label has a glued value (e.g. "Medic:Dr.", "Contract:CJAS"),
        # redact the entire token
        if _has_glued_value(label_anchor['text']):
            parts = label_anchor['text'].split(':', 1)
            value_part = parts[1].strip() if len(parts) > 1 else ""
            if value_part and not _is_safe_by_content(value_part):
                label_anchor['redact'] = True
                label_anchor['reason'] = f'glued_value:{label_kw}'
                print(f"  [Phase2] Glued PII: '{label_anchor['text']}' (label={label_kw})")

        # Scan forward for value words on the SAME ROW, to the RIGHT only.
        last_right_edge = label_anchor['box'][1][0]

        for j in range(scan_start, min(scan_start + 8, len(word_entries))):
            candidate = word_entries[j]

            # Must be on the same horizontal row as the label anchor
            if not _boxes_on_same_row(label_anchor['box'], candidate['box']):
                break

            # Must be to the right
            if not _box_is_right_of(label_anchor['box'], candidate['box']):
                continue

            # Must be within the max horizontal gap
            gap = candidate['box'][0][0] - last_right_edge
            if gap > MAX_LABEL_VALUE_GAP:
                break

            # Stop if we hit another label
            if _extract_pii_label(candidate['text']) is not None:
                break
            mw_label, _ = _extract_multiword_label(word_entries, j)
            if mw_label is not None:
                break
            if _is_safe_label(candidate['text']):
                break

            # Skip if clearly safe medical content
            if _is_safe_by_content(candidate['text']):
                break

            candidate['redact'] = True
            candidate['reason'] = f'value_of:{label_kw}'
            last_right_edge = candidate['box'][1][0]
            print(f"  [Phase2] PII value: '{candidate['text']}' (label={label_kw})")

    # --- Phase 3: Handle "Telefon:" / "Mobil:" below the header --------------
    # These often have the phone number on the same row or on the next line
    # We already handled same-row in Phase 2. For "below" cases, do a targeted
    # scan: if a PII label has no same-row values, check the word just below.
    for i, entry in enumerate(word_entries):
        label_kw = _extract_pii_label(entry['text'])
        if label_kw is None:
            continue
        if _in_clinic_header(entry['box']):
            continue

        # Only certain labels commonly have values on the next line
        if label_kw not in BELOW_VALUE_LABELS:
            continue

        # Check if we already found same-row values
        has_same_row_value = False
        for j in range(i + 1, min(i + 8, len(word_entries))):
            cand = word_entries[j]
            if not _boxes_on_same_row(entry['box'], cand['box']):
                break
            if _box_is_right_of(entry['box'], cand['box']) and cand['redact']:
                has_same_row_value = True
                break

        # If no same-row values were found, check the next word(s) below.
        # Be very conservative: only redact if the candidate genuinely looks
        # like PII data (contains digits for phone/code, or is a multi-word
        # name), NOT if it looks like another form label.
        if not has_same_row_value:
            for j in range(i + 1, min(i + 5, len(word_entries))):
                cand = word_entries[j]

                # Skip words on the same row (already handled by Phase 2)
                if _boxes_on_same_row(entry['box'], cand['box']):
                    continue

                # Check if it's just below (within ~3% vertical distance)
                label_bottom = entry['box'][1][1]
                cand_top = cand['box'][0][1]
                if cand_top > label_bottom + 0.03:
                    break  # too far below

                # Must be roughly in the same horizontal column
                if abs(cand['box'][0][0] - entry['box'][0][0]) > 0.15:
                    break

                if _is_safe_by_content(cand['text']):
                    break

                # Skip if the candidate is itself a PII label
                if _extract_pii_label(cand['text']) is not None:
                    break

                # Skip if the candidate looks like a form label word
                # (check against ALL known label keywords, both PII and safe)
                cand_norm = _normalise(cand['text']).rstrip(':.')
                is_label_word = False
                all_label_words = set()
                for kw in PII_LABEL_KEYWORDS + list(SAFE_LABEL_KEYWORDS):
                    for w in kw.split():
                        all_label_words.add(w)
                if cand_norm in all_label_words:
                    is_label_word = True
                if is_label_word:
                    break

                cand['redact'] = True
                cand['reason'] = f'below_label:{label_kw}'
                print(f"  [Phase3] PII below: '{cand['text']}' (label={label_kw})")

    # --- Phase 4: QR / barcode detection (visual, conservative) --------------
    qr_regions = _detect_qr_barcode_regions(image_bgr)
    print(f"[Anon] Detected {len(qr_regions)} QR/barcode region(s).")

    # --- Phase 5: Redact the "buletin nr." value (document number) -----------
    # Look for "nr." that is the last word in a "Buletin de analize medicale nr."
    # line.  The value (the document number) is the blacked-out region right
    # after "nr." in the original image.
    for i, entry in enumerate(word_entries):
        norm = _normalise(entry['text'])
        if norm != "nr.":
            continue
        if _box_y_center(entry['box']) > 0.25:
            continue

        # Verify context: check that a preceding word is "medicale"/"analize"/"buletin"
        is_buletin_context = False
        for k in range(max(0, i - 6), i):
            prev_norm = _normalise(word_entries[k]['text'])
            if prev_norm in ("medicale", "analize", "buletin"):
                is_buletin_context = True
                break
        if not is_buletin_context:
            continue

        # Also skip if a preceding word is a street name indicator
        is_street_context = False
        for k in range(max(0, i - 3), i):
            prev_norm = _normalise(word_entries[k]['text'])
            if prev_norm.rstrip(',') in ("str", "str.", "motilor", "bd", "bd.",
                                          "calea", "sos", "sos.", "aleea"):
                is_street_context = True
                break
        if is_street_context:
            continue

        for j in range(i + 1, min(i + 3, len(word_entries))):
            cand = word_entries[j]
            if _boxes_on_same_row(entry['box'], cand['box']) and _box_is_right_of(entry['box'], cand['box']):
                # Must look like a number/code, not a regular word
                if re.search(r'\d', cand['text']):
                    cand['redact'] = True
                    cand['reason'] = 'buletin_nr'
                    print(f"  [Phase5] Buletin nr: '{cand['text']}'")

    # --- Apply redaction to image --------------------------------------------
    redacted = image_bgr.copy()

    redacted_count = 0
    for entry in word_entries:
        if entry['redact']:
            (x0, y0), (x1, y1) = entry['box']
            px0 = max(int(x0 * w_img) - margin_px, 0)
            py0 = max(int(y0 * h_img) - margin_px, 0)
            px1 = min(int(x1 * w_img) + margin_px, w_img)
            py1 = min(int(y1 * h_img) + margin_px, h_img)
            cv2.rectangle(redacted, (px0, py0), (px1, py1), redact_color, -1)
            redacted_count += 1

    # Redact QR/barcode regions
    for (x, y, w, h) in qr_regions:
        px0 = max(x - margin_px, 0)
        py0 = max(y - margin_px, 0)
        px1 = min(x + w + margin_px, w_img)
        py1 = min(y + h + margin_px, h_img)
        cv2.rectangle(redacted, (px0, py0), (px1, py1), redact_color, -1)
        redacted_count += 1
        print(f"  [Redact] QR/Barcode region at ({x},{y},{w},{h})")

    print(f"[Anon] Redacted {redacted_count} region(s) total.")

    # --- Debug overlay (optional) --------------------------------------------
    if debug:
        for entry in word_entries:
            if not entry['redact']:
                (x0, y0), (x1, y1) = entry['box']
                px0 = int(x0 * w_img)
                py0 = int(y0 * h_img)
                px1 = int(x1 * w_img)
                py1 = int(y1 * h_img)
                cv2.rectangle(redacted, (px0, py0), (px1, py1), (0, 200, 0), 1)

    # --- Save output ---------------------------------------------------------
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_anon{ext}"

    cv2.imwrite(output_path, redacted)
    print(f"[Anon] Saved anonymised image: {output_path}")
    return output_path


# ========================== CLI ENTRY POINT ================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python anonymisation_module.py <image_path> [output_path] [--debug]")
        sys.exit(1)

    input_path = sys.argv[1]
    debug_mode = "--debug" in sys.argv
    out_path = None
    for arg in sys.argv[2:]:
        if arg != "--debug":
            out_path = arg
            break

    anonymise_image(input_path, out_path, debug=debug_mode)
