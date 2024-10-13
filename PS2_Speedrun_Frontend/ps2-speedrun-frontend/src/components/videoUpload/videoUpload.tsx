// src/components/VideoUpload.tsx
import React from 'react';
import { FileUploaderRegular } from '@uploadcare/react-uploader'; // Import Uploadcare uploader
import '@uploadcare/react-uploader/core.css'; // Import Uploadcare styles

const VideoUpload: React.FC = () => {
    return (
        <div className="container">
            <h1 className="ps2-logo">Upload Your Speedrun Video</h1>

            {/* Uploadcare Upload Button */}
            <FileUploaderRegular
                pubkey="3f039c07fd686d0dd022" // Replace with your Uploadcare public key
                onFileUploadSuccess={(file) => {
                    console.log('File uploaded:', file);  // Just log the file info for now
                }}
            />
        </div>
    );
};

export default VideoUpload;
