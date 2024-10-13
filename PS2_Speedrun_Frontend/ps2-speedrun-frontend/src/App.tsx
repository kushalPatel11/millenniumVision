// src/App.tsx
import React, { useState } from 'react';
import './App.css';
import Login from './components/login';
import VideoUpload from './components/videoUpload';

const App: React.FC = () => {
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  const handleLogin = () => {
    setIsLoggedIn(true);
  };

  return (
    <div className="App">
      {/* {!isLoggedIn ? <Login onLogin={handleLogin} /> : */}
       <VideoUpload />
    </div>
  );
};

export default App;
