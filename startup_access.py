"""Capability gate shared by the gateway and the search process."""


def request_available(state, path, params, method='GET'):
    if path in {'/', '/ready', '/favicon.ico'} or path.startswith(('/static/', '/dres/')):
        return True
    if state.get('ready'):
        return True
    if not state.get('text_ready'):
        return False
    if path == '/search' and method == 'GET':
        # Never silently convert a visual query into an OCR query.
        return (not params.get('query', '').strip() or params.get('search_mode') == 'ocr') and params.get('rerank', '').lower() not in {'true', '1', 'yes', 'on'}
    if path in {'/asr/search', '/asr/frames'} and method == 'POST':
        return True
    if method in {'GET', 'HEAD'}:
        return path in {'/status', '/search/features', '/asr/config', '/asr/status',
                        '/frame-idx', '/frame/timeline', '/frame/neighbors',
                        '/evaluation/status'} or path.startswith(('/image/', '/video/', '/frame/meta/'))
    return path == '/evaluation/event' and method == 'POST'
