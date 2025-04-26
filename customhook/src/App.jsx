import React, { useState } from 'react'
import useCount from './hooks/useCount'

function App() {
  const [count, increment] = useCount()
  return (
    <div style={{display: 'flex', justifyContent: 'center', alignItems: 'center', width: '100vw'}}>
      <button onClick={() => increment(10)}>
        <p>Count: {count}</p>
      </button>
    </div>
  )
}

export default App
