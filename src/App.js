import React, { useState, useEffect } from 'react';
import styled from 'styled-components';

const Container = styled.div`
  width: 100vw;
  height: 100vh;
  position: relative;
  overflow: hidden;
`;

const MapContainer = styled.div`
  width: 100%;
  height: 100vh;
  position: relative;
`;

const LoadingMessage = styled.div`
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  padding: 20px 40px;
  background: rgba(255, 255, 255, 0.9);
  border-radius: 10px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  font-size: 18px;
  color: #4a5568;
  backdrop-filter: blur(10px);
`;

const PromptBox = styled.div`
  position: fixed;
  bottom: 20px;
  left: 50%;
  transform: translateX(-50%);
  width: 60%;
  max-width: 600px;
  padding: 15px;
  background: rgba(255, 255, 255, 0.8);
  backdrop-filter: blur(10px);
  border-radius: 15px;
  box-shadow: 
    0 10px 15px -3px rgba(0, 0, 0, 0.1),
    0 4px 6px -2px rgba(0, 0, 0, 0.05),
    inset 0 -20px 20px -20px rgba(255, 255, 255, 0.3),
    inset 0 20px 20px -20px rgba(255, 255, 255, 0.3);
  border: 1px solid rgba(255, 255, 255, 0.3);
  display: flex;
  gap: 10px;
  z-index: 1000;
`;

const Input = styled.input`
  flex: 1;
  padding: 10px;
  border: none;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.9);
  font-size: 16px;
  box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1);
  &:focus {
    outline: none;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1), 0 0 0 2px rgba(74, 108, 247, 0.2);
  }
`;

const Button = styled.button`
  padding: 10px 20px;
  border: none;
  border-radius: 8px;
  background: linear-gradient(135deg, #6e8efb, #4a6cf7);
  color: white;
  cursor: pointer;
  font-weight: bold;
  transition: all 0.2s;
  &:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  }
  &:disabled {
    opacity: 0.7;
    cursor: not-allowed;
  }
`;

const AnalyticsPanel = styled.div`
  position: fixed;
  top: 20px;
  right: 20px;
  width: 300px;
  max-height: 80vh;
  overflow-y: auto;
  background: rgba(255, 255, 255, 0.92);
  backdrop-filter: blur(12px) saturate(180%);
  border-radius: 15px;
  padding: 20px;
  box-shadow: 
    0 10px 15px -3px rgba(0, 0, 0, 0.1),
    0 4px 6px -2px rgba(0, 0, 0, 0.05),
    inset 0 -20px 20px -20px rgba(255, 255, 255, 0.5),
    inset 0 20px 20px -20px rgba(255, 255, 255, 0.5);
  border: 1px solid rgba(255, 255, 255, 0.4);
  z-index: 1000;
  transition: all 0.3s ease;

  &::-webkit-scrollbar {
    width: 8px;
    background: transparent;
  }

  &::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    margin: 4px;
  }

  &::-webkit-scrollbar-thumb {
    background: rgba(74, 108, 247, 0.2);
    border-radius: 4px;
    border: 2px solid transparent;
    background-clip: padding-box;

    &:hover {
      background: rgba(74, 108, 247, 0.3);
      border: 2px solid transparent;
      background-clip: padding-box;
    }
  }

  &:hover {
    box-shadow: 
      0 15px 20px -3px rgba(0, 0, 0, 0.12),
      0 6px 8px -2px rgba(0, 0, 0, 0.06),
      inset 0 -20px 20px -20px rgba(255, 255, 255, 0.6),
      inset 0 20px 20px -20px rgba(255, 255, 255, 0.6);
  }

  img {
    display: block;
    width: 100%;
    max-width: 100%;
    height: auto;
    margin: 15px 0;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    transition: transform 0.2s ease;

    &:hover {
      transform: scale(1.02);
      box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }
  }

  h3 {
    color: #2d3748;
    margin: 0 0 10px 0;
    font-size: 1.2em;
    font-weight: 600;
  }

  p {
    color: #4a5568;
    line-height: 1.5;
    margin: 0 0 15px 0;
  }

  ul {
    color: #4a5568;
    padding-left: 20px;
    margin: 0;
    
    li {
      margin-bottom: 5px;
      line-height: 1.4;
    }
  }
`;

function App() {
  const [prompt, setPrompt] = useState('');
  const [loading, setLoading] = useState(false);
  const [mapHTML, setMapHTML] = useState("http://127.0.0.1:5000/maps/filtered_map.html");
  const [mapLoading, setMapLoading] = useState(false);
  const [mapError, setMapError] = useState(null);
  const [analytics, setAnalytics] = useState({
    summary: '',
    count: 0,
    logs: [],
    visualization: null
  });

  const handlePromptSubmit = async () => {
    if (!prompt.trim()) return;
    
    setLoading(true);
    setMapLoading(true);
    setMapError(null);
    
    try {
      const response = await fetch('http://127.0.0.1:5000/process-prompt', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt: prompt.trim() })
      });

      const data = await response.json();
      
      if (data.error) {
        console.error('Error:', data.error);
        setMapError(data.error);
        setLoading(false);
        setMapLoading(false);
        return;
      }

      setAnalytics({
        summary: data.summary,
        count: data.count,
        logs: data.logs,
        visualization: data.visualization
      });

      // Wait a short moment to ensure the map file is ready
      await new Promise(resolve => setTimeout(resolve, 500));

      // Update map URL with timestamp to force refresh
      const timestamp = Date.now();
      const newMapUrl = `http://127.0.0.1:5000/maps/${data.filteredMap}?t=${timestamp}`;
      setMapHTML(newMapUrl);
      
    } catch (error) {
      console.error('Error:', error);
      setMapError('Failed to process request');
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handlePromptSubmit();
    }
  };

  return (
    <Container>
      <MapContainer>
        <iframe
          key={mapHTML}
          src={mapHTML}
          title="Interactive Map"
          style={{
            width: "100%",
            height: "100vh",
            border: "none",
            display: mapLoading ? "none" : "block"
          }}
          onLoad={() => {
            console.log("Map iframe loaded");
            setMapLoading(false);
          }}
          onError={(e) => {
            console.error("Map iframe error:", e);
            setMapError('Failed to display map');
            setMapLoading(false);
          }}
        />
        {mapLoading && (
          <LoadingMessage>Loading map...</LoadingMessage>
        )}
        {mapError && (
          <LoadingMessage style={{ color: '#e53e3e' }}>{mapError}</LoadingMessage>
        )}
      </MapContainer>

      <PromptBox>
        <Input
          type="text"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Enter your search query..."
          disabled={loading}
        />
        <Button onClick={handlePromptSubmit} disabled={loading}>
          {loading ? 'Searching...' : 'Search'}
        </Button>
      </PromptBox>
      
      <AnalyticsPanel>
        {analytics.summary && (
          <>
            <h3 style={{ margin: '0 0 10px 0', color: '#2d3748' }}>Summary</h3>
            <p style={{ color: '#4a5568' }}>{analytics.summary}</p>
          </>
        )}
        
        {analytics.visualization && (
          <div style={{ marginTop: '20px' }}>
            <img 
              src={`data:image/png;base64,${analytics.visualization.data}`}
              alt={analytics.visualization.title || "Data Visualization"}
            />
          </div>
        )}
        
        {analytics.count > 0 && (
          <p style={{ color: '#4a5568' }}>
            <strong>Results:</strong> {analytics.count} matches found
          </p>
        )}

        {analytics.logs.length > 0 && (
          <>
            <h3 style={{ margin: '15px 0 10px 0', color: '#2d3748' }}>Processing Steps</h3>
            <ul style={{ 
              color: '#4a5568', 
              paddingLeft: '20px',
              margin: '0'
            }}>
              {analytics.logs.map((log, index) => (
                <li key={index} style={{ marginBottom: '5px' }}>{log}</li>
              ))}
            </ul>
          </>
        )}
      </AnalyticsPanel>
    </Container>
  );
}

export default App;