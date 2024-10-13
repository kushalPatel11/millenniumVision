// src/App.tsx
import React, { useState } from 'react';
import './App.css';
import './tailwind.css';
import { SmoothScrollHero } from './components/SmoothScrollHero';
import VideoUpload from './components/videoUpload/videoUpload';

const App: React.FC = () => {
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  const handleLogin = () => {
    setIsLoggedIn(true);
  };

  return (
    <div className="App">
      {!isLoggedIn ? (
        <SmoothScrollHero onLogin={handleLogin} />  /* Pass the handleLogin function */
      ) : (
        <VideoUpload />
      )
      }
    </div>
  );
};

export default App;
