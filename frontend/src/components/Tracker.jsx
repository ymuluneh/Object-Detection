import React, { useRef, useState, useEffect } from "react";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import "@tensorflow/tfjs";
import axios from "axios";
import './Tracker.css'; // Import the CSS file

// Simple centroid tracker
class CentroidTracker {
  constructor(maxDisappeared = 50, maxDistance = 50) {
    this.nextObjectID = 1;
    this.objects = new Map(); // id -> centroid
    this.disappeared = new Map(); // id -> count
    this.maxDisappeared = maxDisappeared;
    this.maxDistance = maxDistance;
  }

  register(centroid) {
    const id = this.nextObjectID++;
    this.objects.set(id, centroid);
    this.disappeared.set(id, 0);
    return id;
  }

  deregister(id) {
    this.objects.delete(id);
    this.disappeared.delete(id);
  }

  update(rects) {
    const inputCentroids = rects.map((r) => {
      const cx = Math.round((r[0] + r[2]) / 2);
      const cy = Math.round((r[1] + r[3]) / 2);
      return [cx, cy];
    });

    if (this.objects.size === 0) {
      rects.forEach((r) => {
        const cx = Math.round((r[0] + r[2]) / 2);
        const cy = Math.round((r[1] + r[3]) / 2);
        this.register([cx, cy]);
      });
      return this.objects;
    }

    if (rects.length === 0) {
      for (const id of Array.from(this.disappeared.keys())) {
        this.disappeared.set(id, this.disappeared.get(id) + 1);
        if (this.disappeared.get(id) > this.maxDisappeared) {
          this.deregister(id);
        }
      }
      return this.objects;
    }

    const objectIDs = Array.from(this.objects.keys());
    const objectCentroids = Array.from(this.objects.values());

    const D = [];
    for (let i = 0; i < objectCentroids.length; i++) {
      const row = [];
      for (let j = 0; j < inputCentroids.length; j++) {
        const dx = objectCentroids[i][0] - inputCentroids[j][0];
        const dy = objectCentroids[i][1] - inputCentroids[j][1];
        row.push(Math.hypot(dx, dy));
      }
      D.push(row);
    }

    const assignedRows = new Set();
    const assignedCols = new Set();
    const matches = [];

    while (true) {
      let minVal = Infinity;
      let minR = -1,
        minC = -1;
      for (let r = 0; r < D.length; r++) {
        if (assignedRows.has(r)) continue;
        for (let c = 0; c < D[r].length; c++) {
          if (assignedCols.has(c)) continue;
          if (D[r][c] < minVal) {
            minVal = D[r][c];
            minR = r;
            minC = c;
          }
        }
      }
      if (minVal === Infinity) break;
      if (minVal > this.maxDistance) break;
      assignedRows.add(minR);
      assignedCols.add(minC);
      matches.push([minR, minC]);
    }

    const unmatchedRows = new Set(Array.from(Array(D.length).keys()));
    const unmatchedCols = new Set(
      Array.from(Array(inputCentroids.length).keys())
    );
    matches.forEach(([r, c]) => {
      unmatchedRows.delete(r);
      unmatchedCols.delete(c);
      const objectID = objectIDs[r];
      this.objects.set(objectID, inputCentroids[c]);
      this.disappeared.set(objectID, 0);
    });

    for (const r of unmatchedRows) {
      const objectID = objectIDs[r];
      this.disappeared.set(objectID, this.disappeared.get(objectID) + 1);
      if (this.disappeared.get(objectID) > this.maxDisappeared) {
        this.deregister(objectID);
      }
    }

    for (const c of unmatchedCols) {
      this.register(inputCentroids[c]);
    }

    return this.objects;
  }
}

export default function Tracker() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const modelRef = useRef(null);
  const rafRef = useRef(null);
  const [running, setRunning] = useState(false);
  const [logEnabled, setLogEnabled] = useState(false);
  const [fps, setFps] = useState(0);
  const [filterClass, setFilterClass] = useState("");
  const [videoFile, setVideoFile] = useState(null);
  const [uploading, setUploading] = useState(false);

  const tracker = useRef(new CentroidTracker(40, 80));
  const frameIdx = useRef(0);

  useEffect(() => {
    return () => {
      stop();
    };
  }, []);

  async function loadModel() {
    if (!modelRef.current) {
      modelRef.current = await cocoSsd.load();
      console.log("Model loaded");
    }
  }

  async function start() {
    await loadModel();
    setRunning(true);
    frameIdx.current = 0;

    if (videoFile) {
      const videoUrl = URL.createObjectURL(videoFile);
      videoRef.current.src = videoUrl;
      videoRef.current.loop = true;
      videoRef.current.onended = () => {
        stop();
        URL.revokeObjectURL(videoUrl);
      };
      await videoRef.current.play();
    } else {
      const constraints = {
        video: { width: 640, height: 480, facingMode: "environment" },
        audio: false,
      };
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
    }
    
    tracker.current = new CentroidTracker(40, 80);
    tick();
  }

  function stop() {
    setRunning(false);
    if (rafRef.current) cancelAnimationFrame(rafRef.current);

    if (videoRef.current) {
      if (videoRef.current.srcObject) {
        const tracks = videoRef.current.srcObject.getTracks();
        tracks.forEach((t) => t.stop());
        videoRef.current.srcObject = null;
      } else if (videoRef.current.src) {
        videoRef.current.pause();
        URL.revokeObjectURL(videoRef.current.src);
        videoRef.current.src = "";
      }
    }
  }

  async function tick() {
    const t0 = performance.now();
    if (!videoRef.current || videoRef.current.readyState < 2) {
      rafRef.current = requestAnimationFrame(tick);
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const predictions = await modelRef.current.detect(video);

    const wanted = filterClass
      .split(",")
      .map((s) => s.trim().toLowerCase())
      .filter(Boolean);
    const filtered = wanted.length
      ? predictions.filter((p) => wanted.includes(p.class.toLowerCase()))
      : predictions;

    const rects = filtered.map((p) => {
      const [x, y, w, h] = p.bbox;
      return [
        Math.round(x),
        Math.round(y),
        Math.round(x + w),
        Math.round(y + h),
      ];
    });

    const objects = tracker.current.update(rects);

    const detsWithIds = [];
    for (let i = 0; i < filtered.length; i++) {
      const p = filtered[i];
      const rect = rects[i];
      const cx = Math.round((rect[0] + rect[2]) / 2);
      const cy = Math.round((rect[1] + rect[3]) / 2);
      let bestId = null;
      let bestDist = Infinity;
      for (const [id, cent] of objects.entries()) {
        const d = Math.hypot(cent[0] - cx, cent[1] - cy);
        if (d < bestDist) {
          bestDist = d;
          bestId = id;
        }
      }
      detsWithIds.push({
        track_id: bestId,
        class_name: p.class,
        conf: p.score,
        x1: rect[0],
        y1: rect[1],
        x2: rect[2],
        y2: rect[3],
        frame_index: frameIdx.current,
      });
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 2;
    ctx.font = "16px Arial";
    for (const d of detsWithIds) {
      ctx.strokeStyle = "rgba(0,255,0,0.9)";
      ctx.fillStyle = "rgba(0,255,0,0.2)";
      ctx.beginPath();
      ctx.rect(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1);
      ctx.stroke();
      const text = `ID ${d.track_id ?? "-"} | ${d.class_name} ${(
        d.conf * 100
      ).toFixed(1)}%`;
      const textW = ctx.measureText(text).width;
      ctx.fillRect(d.x1, d.y1 - 20, textW + 8, 20);
      ctx.fillStyle = "#000";
      ctx.fillText(text, d.x1 + 4, d.y1 - 5);
    }

    if (logEnabled && detsWithIds.length) {
      try {
        await axios.post("/api/detections", { detections: detsWithIds });
      } catch (err) {
        console.warn("Failed to send logs", err.message);
      }
    }

    const t1 = performance.now();
    const frameTime = t1 - t0;
    setFps((prev) => 0.9 * prev + 0.1 * (1000 / frameTime));

    frameIdx.current += 1;
    rafRef.current = requestAnimationFrame(tick);
  }

  function handleFileChange(event) {
    const file = event.target.files[0];
    setVideoFile(file);
    if (running) {
      stop();
    }
  }
  
  async function uploadVideo() {
    if (!videoFile) {
      alert('Please select a video file first.');
      return;
    }
    setUploading(true);
    const formData = new FormData();
    formData.append('videoFile', videoFile);

    try {
      const response = await axios.post('/api/upload-video', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      console.log('Upload success:', response.data);
      alert('Video file uploaded successfully!');
    } catch (error) {
      console.error('Upload failed:', error);
      alert('Failed to upload video file.');
    } finally {
      setUploading(false);
    }
  }

  function handleVideoControl(action) {
    if (videoRef.current) {
      switch (action) {
        case 'pause':
          videoRef.current.pause();
          break;
        case 'play':
          videoRef.current.play();
          break;
        case 'forward':
          videoRef.current.currentTime += 10;
          break;
        case 'backward':
          videoRef.current.currentTime -= 10;
          break;
        default:
          break;
      }
    }
  }

  return (
    <div className="app-container">
      {/* <h1>Object Tracking Application</h1> */}
      <div className="controls-panel">
        <div className="control-group">
          <button className="control-button" onClick={() => (running ? stop() : start())}>
            {running ? "Stop" : videoFile ? "Start Video" : "Start Camera"}
          </button>
          <label className="file-input-label">
            <input className="file-input" type="file" accept="video/*" onChange={handleFileChange} />
            Upload Video
          </label>
          
          {videoFile && (
            <button className="control-button" onClick={uploadVideo} disabled={uploading}>
              {uploading ? 'Uploading...' : 'Send File to Backend'}
            </button>
          )}
        </div>
        
        {videoFile && (
          <div className="video-controls-group">
            <button className="control-button" onClick={() => handleVideoControl('pause')}>Pause</button>
            <button className="control-button" onClick={() => handleVideoControl('play')}>Play</button>
            <button className="control-button" onClick={() => handleVideoControl('backward')}>&lt;&lt; 10s</button>
            <button className="control-button" onClick={() => handleVideoControl('forward')}>&gt;&gt; 10s</button>
          </div>
        )}

        <div className="settings-group">
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={logEnabled}
              onChange={(e) => setLogEnabled(e.target.checked)}
            />
            Send logs to backend
          </label>
          <label className="filter-label">
            Filter classes (comma):
            <input
              className="filter-input"
              value={filterClass}
              onChange={(e) => setFilterClass(e.target.value)}
              placeholder="e.g. person,car"
            />
          </label>
          <div className="fps-display">FPS: {fps.toFixed(1)}</div>
        </div>
      </div>

      <div className="video-and-canvas-container">
        <video
          ref={videoRef}
          className="video-element"
          onCanPlay={() => {
            if (!running) {
              videoRef.current.play();
            }
          }}
          muted
          playsInline
        ></video>
        <canvas
          ref={canvasRef}
          className="canvas-element"
        ></canvas>
      </div>
    </div>
  );
}