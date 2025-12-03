function App() {
  return (
    <div style={{ 
      background: 'red', 
      color: 'white', 
      padding: '50px',
      fontSize: '30px'
    }}>
      <h1>HELLO! CAN YOU SEE THIS?</h1>
      <p>If you see this RED screen, React is working!</p>
      <p>Current time: {new Date().toLocaleTimeString()}</p>
    </div>
  );
}

export default App;