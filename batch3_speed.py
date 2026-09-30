"""Explicit speed profiles; exported images and frame sampling are unchanged."""
PROFILES = {
    'legacy': {},
    'a100-fast': {'OCR_MAX_SIDE': '1280', 'OCR_DETECT_BATCH_SIZE': '8',
                  'OCR_RECOGNITION_BATCH_SIZE': '64', 'SKIP_DETECTION_BACKFILL': '1'},
}


def profile(args):
    return getattr(args, 'speed_profile', 'legacy')


def environment(args):
    return {'OCR_MAX_SIDE': '0', 'OCR_DETECT_BATCH_SIZE': str(args.ocr_batch_size),
            'OCR_RECOGNITION_BATCH_SIZE': '0', 'SKIP_DETECTION_BACKFILL': '0',
            **PROFILES[profile(args)]}
