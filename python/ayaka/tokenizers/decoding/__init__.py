"""Turning generated token ids back into text, one request at a time.

``incremental``  the detokenizer base class and the stream and slow paths.
``byte``         the fast path over a verified token-to-bytes table.
``stop``         stop-string matching shared by every path.
"""
