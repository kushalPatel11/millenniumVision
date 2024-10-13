// src/components/Login.tsx
import React, { useState } from 'react';
import axios from 'axios';

interface LoginProps {
    onLogin: () => void;
}

const Login: React.FC<LoginProps> = ({ onLogin }) => {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');

    const handleLogin = async (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        try {
            const response = await axios.post('http://localhost:5000/api/login', { username, password });
            if (response.data.success) {
                onLogin();
            } else {
                setError('Invalid credentials');
            }
        } catch (err) {
            setError('Error during login');
        }
    };

    return (
        <div className="container">
            <h1 className="ps2-logo">PS2 Speedrun Login</h1>
            <form onSubmit={handleLogin}>
                <input
                    type="text"
                    placeholder="Username"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                />
                <br />
                <input
                    type="password"
                    placeholder="Password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                />
                <br />
                <button className="button" type="submit">Login</button>
                {error && <p style={{ color: 'red' }}>{error}</p>}
            </form>
        </div>
    );
};

export default Login;
