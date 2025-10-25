"""
Integration tests for the genomic annotation system
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    """Test basic health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "version" in data

def test_liftover_single():
    """Test single coordinate liftover"""
    response = client.post(
        "/liftover/single",
        params={
            "chrom": "chr17",
            "pos": 41196312,
            "from_build": "hg19",
            "to_build": "hg38"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data.get("success") is True
    assert "lifted_pos" in data

def test_validation_report():
    """Test validation report generation"""
    response = client.get("/validation-report")
    assert response.status_code in [200, 503]  # 503 if services not loaded

if __name__ == "__main__":
    pytest.main([__file__, "-v"])