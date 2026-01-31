
from fastapi.testclient import TestClient
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch
import os

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# We need to ensure we can import the api even if model is not ready, 
# but main.py imports GestureDetector which imports model.
# If training is not done, main.py might fail startup if we are not careful.
# But main.py handles exception in load_model.

from src.api.main import app

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_read_root(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])

    def test_static_files(self):
        # Ensure static files are mounted
        response = self.client.get("/static/style.css")
        # It might be 200 or 404 depending if file exists, but let's check status
        # If file exists it should be 200.
        if (PROJECT_ROOT / "src/api/static/style.css").exists():
            self.assertEqual(response.status_code, 200)

    @patch('src.inference.detector.GestureDetector.predict')
    def test_predict_mocked(self, mock_predict):
        # Mock prediction to return a dummy label
        mock_predict.return_value = ("gesture_0", 0.95)
        
        # We need to make sure the app thinks the detector is loaded.
        # Check if detector is loaded in app
        from src.api import main
        
        # If detector is None (real loading failed or logic handled it), we inject a mock
        if main.detector is None:
            main.detector = MagicMock()
            main.detector.predict = mock_predict
            main.detector.transform = MagicMock()
            main.detector.model = MagicMock()
            main.detector.device = 'cpu'
            main.detector.classes = ['gesture_0', 'gesture_1', 'gesture_2']
        
        # Create a valid minimal 1x1 PNG image
        valid_png_bytes = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
        
        files = {'file': ('test.png', valid_png_bytes, 'image/png')}
        response = self.client.post("/predict", files=files)
        
        # Since we are mocking the internal logic of predict_image in main.py 
        # heavily relies on detector. 
        # main.py code:
        # input_tensor = detector.transform(image).unsqueeze(0).to(detector.device)
        # outputs = detector.model(input_tensor)
        
        # My mock above for detector.predict is NOT used in main.py because main.py 
        # calls detector.transform and detector.model directly!
        # See main.py code I read earlier.
        
        # So I need to mock detector.transform and detector.model
        main.detector.transform.return_value = MagicMock()
        main.detector.transform.return_value.unsqueeze.return_value.to.return_value = MagicMock()
        
        # Mock model output
        # Model returns logits. 
        # We need something that torch.nn.functional.softmax can handle
        import torch
        mock_output = torch.tensor([[0.1, 0.8, 0.1]]) # 3 classes
        main.detector.model.return_value = mock_output
        main.detector.classes = ['gesture_0', 'gesture_1', 'gesture_2']
        
        response = self.client.post("/predict", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['label'], 'gesture_1')

if __name__ == '__main__':
    unittest.main()
