# VisionGuard AI (Step 2 + 3)

Run:
    python run_pipeline.py data\videos\demo.mp4 --force     (optional, the app can do this itself)
    python app.py                                          (opens http://127.0.0.1:8000)

Optional ground truth for the Evaluation page: data\videos\demo.gt.json
    { "events": [ {"type":"loiter","start":70,"end":95}, {"type":"abandoned","start":110,"end":140}, {"type":"running","start":130,"end":135} ] }
