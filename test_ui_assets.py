"""Verify the independently deployable UI assets without running model startup."""
import re
import unittest
from fastapi.testclient import TestClient
import main


class UIAssetsTests(unittest.TestCase):
    def test_page_assets_are_served(self):
        client = TestClient(main.app)
        response = client.get('/')
        self.assertEqual(response.status_code, 200)
        assets = re.findall(r'(?:src|href)="(static/[^\"]+)"', response.text)
        self.assertEqual(len(assets), 5)
        for asset in assets:
            with self.subTest(asset=asset):
                result = client.get('/' + asset)
                self.assertEqual(result.status_code, 200)
                self.assertIn('text/', result.headers['content-type'])
                self.assertTrue(result.headers.get('etag'))

    def test_static_route_does_not_expose_source(self):
        client = TestClient(main.app)
        for route in ('/static/main.py', '/static/%2e%2e/main.py', '/static/missing.js'):
            with self.subTest(route=route):
                self.assertEqual(client.get(route).status_code, 404)
