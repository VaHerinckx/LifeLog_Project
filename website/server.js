import express from 'express';
import cors from 'cors';
import axios from 'axios';
import path from 'path';
import { fileURLToPath } from 'url';
import rateLimit from 'express-rate-limit';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();

// HTTP Basic Auth with role-based access
// AUTH_USER/AUTH_PASS = full access
// GUEST_USER/GUEST_PASS = limited access (reading, movies, podcasts, shows only)
const users = {};
if (process.env.AUTH_USER && process.env.AUTH_PASS) {
    users[process.env.AUTH_USER] = { pass: process.env.AUTH_PASS, role: 'admin' };
}
if (process.env.GUEST_USER && process.env.GUEST_PASS) {
    users[process.env.GUEST_USER] = { pass: process.env.GUEST_PASS, role: 'guest' };
}

// Pages allowed for guest users
const GUEST_ALLOWED_PAGES = ['/', '/reading', '/movies', '/podcasts', '/shows'];

if (process.env.NODE_ENV === 'production' && Object.keys(users).length > 0) {
    app.use((req, res, next) => {
        const authHeader = req.headers.authorization;

        if (!authHeader || !authHeader.startsWith('Basic ')) {
            res.setHeader('WWW-Authenticate', 'Basic realm="LifeLog Dashboard"');
            return res.status(401).send('Authentication required');
        }

        const base64Credentials = authHeader.split(' ')[1];
        const credentials = Buffer.from(base64Credentials, 'base64').toString('utf8');
        const [username, password] = credentials.split(':');

        const user = users[username];
        if (user && user.pass === password) {
            req.userRole = user.role;
            return next();
        }

        res.setHeader('WWW-Authenticate', 'Basic realm="LifeLog Dashboard"');
        return res.status(401).send('Invalid credentials');
    });

    // Role-based page access (check before serving static files)
    app.use((req, res, next) => {
        // Skip API routes - they're handled separately
        if (req.path.startsWith('/api/')) {
            return next();
        }

        // Admin has full access
        if (req.userRole === 'admin') {
            return next();
        }

        // Guest: check if page is allowed
        if (req.userRole === 'guest') {
            const isAllowed = GUEST_ALLOWED_PAGES.some(page =>
                req.path === page || req.path.startsWith(page + '/')
            );
            // Also allow static assets (js, css, images, fonts)
            const isStaticAsset = /\.(js|css|png|jpg|jpeg|gif|svg|ico|woff|woff2|ttf|eot)$/i.test(req.path);

            if (isAllowed || isStaticAsset) {
                return next();
            }

            return res.status(403).send('Access denied - this page is not available for guest accounts');
        }

        next();
    });
}

// CORS configuration - restrict to allowed origin in production
const corsOptions = {
  origin: process.env.ALLOWED_ORIGIN || 'http://localhost:5173',
  credentials: true
};
app.use(cors(corsOptions));

// Rate limiting for API endpoints
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // limit each IP to 100 requests per windowMs
  message: 'Too many requests, please try again later.'
});
app.use('/api/', limiter);

// API endpoint to get current user role and allowed pages
app.get('/api/user-info', (req, res) => {
    // In development, return admin role
    if (process.env.NODE_ENV !== 'production') {
        return res.json({ role: 'admin', allowedPages: null });
    }

    // Return guest restrictions
    if (req.userRole === 'guest') {
        return res.json({
            role: 'guest',
            allowedPages: GUEST_ALLOWED_PAGES
        });
    }

    // Admin has full access
    return res.json({ role: 'admin', allowedPages: null });
});

// API endpoint for Google Drive files
app.get('/api/google-drive/:fileId', async (req, res) => {
    try {
        const fileId = req.params.fileId;

        // Only log in development
        if (process.env.NODE_ENV !== 'production') {
            console.log(`Fetching Google Drive file: ${fileId}`);
        }

        const response = await axios({
            url: `https://drive.google.com/uc?export=download&id=${fileId}`,
            method: 'GET',
            responseType: 'text',
            timeout: 60000, // 60 seconds timeout
            maxContentLength: 100 * 1024 * 1024, // 100MB max
            maxBodyLength: 100 * 1024 * 1024 // 100MB max
        });

        if (process.env.NODE_ENV !== 'production') {
            console.log(`Successfully fetched file ${fileId}, size: ${response.data.length} characters`);
        }
        res.send(response.data);
    } catch (error) {
        console.error(`Error fetching file:`, error.message);
        if (error.code === 'ECONNABORTED') {
            res.status(504).send('Request timeout - file too large or connection slow');
        } else if (error.response) {
            res.status(error.response.status).send(error.response.data);
        } else {
            res.status(500).send(error.message);
        }
    }
});

// Serve static files in production
if (process.env.NODE_ENV === 'production') {
    app.use(express.static(path.join(__dirname, 'dist')));

    // Handle React Router - serve index.html for all non-API routes
    app.get('*', (req, res, next) => {
        if (req.path.startsWith('/api/')) {
            return next();
        }
        res.sendFile(path.join(__dirname, 'dist', 'index.html'));
    });
}

const PORT = process.env.PORT || 3001;
app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server running on port ${PORT}`);
});
