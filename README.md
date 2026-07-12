# RoadVision AI — Infrastructure Damage Assessment

## Features

- **Real-time detection** of Potholes, Cracks, and Manholes via trained custom YOLOv11 weights
- **Mint-fresh glassmorphic UI** with dynamic Enter key activation
- **Multi-upload memory-safe** file handling with automatic cache clearing
- **Road Safety Index** calculation with severity classification
- **Serverless-ready** architecture optimized for Vercel deployment

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python app.py
```

**Requirements:**
- `best.pt` model file in project root
- `vercel.json` configured for serverless routing
- `requirements.txt` with optimized packages

## API Endpoints

- `GET /` — Dashboard interface
- `POST /api/detect` — Image analysis with YOLOv11 inference
- `GET /api/health` — System health check

## Tech Stack

- **Backend**: Flask, YOLOv11, OpenCV-headless, NumPy
- **Frontend**: TailwindCSS, Vanilla JavaScript
- **Deployment**: Vercel Serverless Functions

---

Engineered by Muhammad Faizan
