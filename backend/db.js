import mysql from 'mysql2/promise';
import dotenv from 'dotenv';
dotenv.config();

const pool = mysql.createPool({
  host: process.env.DB_HOST || "localhost",
  user: process.env.DB_USER || "object_tracking_db",
  password: process.env.DB_PASS || "object_tracking_db",
  database: process.env.DB_NAME || "object_tracking_db",
  waitForConnections: true,
  connectionLimit: 10,
  queueLimit: 0,
});

export default pool;
