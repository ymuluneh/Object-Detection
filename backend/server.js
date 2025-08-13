import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import multer from "multer"; 
import path from "path";
import pool from "./db.js";

dotenv.config();
const app = express();
app.use(cors());
app.use(express.json({ limit: "5mb" }));

// Multer storage configuration
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, "uploads/"); 
  },
  filename: (req, file, cb) => {
    // Generate a unique filename with the original extension
    cb(
      null,
      file.fieldname + "-" + Date.now() + path.extname(file.originalname)
    );
  },
});

const upload = multer({ storage: storage });

//create data base connection
pool
  .getConnection()
  .then((conn) => {
    console.log("Database connected");
    conn.release();
  })
  .catch((err) => {
    console.error("Database connection failed:", err);
  });

// Health check endpoint
app.get("/api/health", (req, res) => res.json({ ok: true }));

// **New endpoint to handle video file upload**
app.post("/api/upload-video", upload.single("videoFile"), (req, res) => {
  if (!req.file) {
    return res.status(400).send("No file uploaded.");
  }
  console.log("Video file saved:", req.file.filename);
  res.status(200).json({
    message: "File uploaded successfully",
    filename: req.file.filename,
    filePath: req.file.path,
  });
});

// Detections API endpoint (unchanged)
app.post("/api/detections", async (req, res) => {
  const { detections } = req.body;
  if (!Array.isArray(detections)) {
    return res.status(400).json({ error: "detections must be an array" });
  }

  const conn = await pool.getConnection();
  try {
    await conn.beginTransaction();
    const insertSql =
      "INSERT INTO detections (track_id, class_name, conf, x1, y1, x2, y2, frame_index) VALUES ?";
    const values = detections.map((d) => [
      d.track_id ?? null,
      d.class_name ?? null,
      typeof d.conf === "number" ? d.conf : parseFloat(d.conf) || 0,
      d.x1 | 0,
      d.y1 | 0,
      d.x2 | 0,
      d.y2 | 0,
      d.frame_index | 0,
    ]);

    if (values.length) {
      await conn.query(insertSql, [values]);
    }

    await conn.commit();
    res.json({ inserted: values.length });
  } catch (err) {
    await conn.rollback();
    console.error(err);
    res.status(500).json({ error: "db_error", details: err.message });
  } finally {
    conn.release();
  }
});

// Simple retrieval endpoint (with limit)
app.get("/api/detections", async (req, res) => {
  const limit = Math.min(parseInt(req.query.limit || "100", 10), 1000);
  const [rows] = await pool.query(
    "SELECT * FROM detections ORDER BY id DESC LIMIT ?",
    [limit]
  );
  res.json(rows);
});

const port = process.env.PORT || 4000;
app.listen(port, () =>
  console.log(`🚀 Backend listening on http://localhost:${port}`)
);
