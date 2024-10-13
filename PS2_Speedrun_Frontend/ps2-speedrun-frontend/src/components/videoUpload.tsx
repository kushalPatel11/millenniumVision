// src/components/VideoUpload.tsx
import React, { useState } from 'react';
import axios from 'axios';

interface VideoResult {
    message: string;
    currentRun: number;
    personalBest: number;
}

const VideoUpload: React.FC = () => {
    const [file, setFile] = useState<File | null>(null);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState<VideoResult | null>(null);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files) {
            setFile(e.target.files[0]);
        }
    };

    const handleUpload = async () => {
        if (!file) return;
        setLoading(true);
        const formData = new FormData();
        formData.append('video', file);
        formData.append('userId', 'guest'); // Set as guest for now

        try {
            const response = await axios.post('http://localhost:5000/api/videos/upload', formData);
            setResult(response.data);
        } catch (err) {
            console.error('Error uploading video', err);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="container">
            <h1 className="ps2-logo">Upload Your Speedrun Video</h1>
            <input type="file" onChange={handleFileChange} />
            <br />
            <button className="button" onClick={handleUpload}>Upload Video</button>
            {loading && <div className="loading-icon"></div>}
            {result && (
                <div>
                    <h2>Video Processed!</h2>
                    <p>{result.message}</p>
                    <p>Current Run: {result.currentRun} seconds</p>
                    <p>Personal Best: {result.personalBest} seconds</p>
                </div>
            )}
        </div>
    );
};

export default VideoUpload;
