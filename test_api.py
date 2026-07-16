import unittest
from fastapi.testclient import TestClient
from api.main import app 

class TestBookRecAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_content_recommendation(self):
        """Content-Based (TF-IDF) Test"""
        response = self.client.post("/recommend/content", json={
            "book_title": "The Hobbit",
            "top_n": 3
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["method"], "Content-Based (TF-IDF)")
        self.assertTrue(len(data["results"]) <= 3)

    def test_semantic_recommendation(self):
        """Deep Learning (FAISS Semantic) Test"""
        response = self.client.post("/recommend/semantic", json={
            "query_text": "The Hobbit",
            "top_n": 3
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["method"], "Deep Learning (FAISS Semantic)")
        self.assertTrue(len(data["results"]) <= 3)

    def test_user_blend_recommendation(self):
        """User Blend - FAISS Test"""
        response = self.client.post("/recommend/user-blend", json={
            "favorite_books": [
                "The Hobbit"
            ],
            "top_n": 3
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertTrue(len(data["results"]) <= 3)

    def test_book_not_found_edge_case(self):
        response = self.client.post("/recommend/content", json={
            "book_title": "This Book Does Absolutely Not Exist In Database 12345",
            "top_n": 3
        })
        self.assertEqual(response.status_code, 404)

if __name__ == "__main__":
    unittest.main()