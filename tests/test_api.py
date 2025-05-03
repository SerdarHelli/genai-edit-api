import pytest
from fastapi.testclient import TestClient
from PIL import Image
import io
from app.main import app

client = TestClient(app)

def create_test_image() -> bytes:
    """Generate a simple RGB image in memory."""
    image = Image.new("RGB", (64, 64), color="blue")
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)
    return buf.getvalue()

@pytest.mark.parametrize("endpoint", [
    "/api/v1/edit/level1",
    "/api/v2/edit/level2"
])
def test_level1_and_level2(endpoint):
    image_bytes = create_test_image()
    files = {"image": ("test.jpg", image_bytes, "image/jpeg")}
    data = {"similarity_level": 0.8}
    if "level2" in endpoint:
        data["prompt"] = "add a scratch"

    response = client.post(endpoint, files=files, data=data)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"

def test_level3():
    image_bytes = create_test_image()
    files = {
        "baseline": ("baseline.jpg", image_bytes, "image/jpeg"),
        "annotated": ("annotated.jpg", image_bytes, "image/jpeg")
    }
    data = {"similarity_level": 0.8}
    response = client.post("/api/v3/edit/level3", files=files, data=data)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
