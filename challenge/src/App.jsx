import React, { useState } from 'react'

function App() {
  const [counter, setCounter] = useState(0)
  const [val, setVal] = useState(0)
  const handleCount = (e) => {
    e.preventDefault()
    setVal(e.target.value)
    console.log('e.target.value: ', e.target.value);
    
  }
  return (
    <div className='flex w-screen h-screen bg-[#242424] text-white justify-center items-center'>
      <div className='flex flex-col gap-8 p-4 w-192 h-96'>
        <div className='flex gap-8'>
          <h2 className='text-3xl font-bold'>Counter Value: </h2>
          <h2 className='text-3xl font-bold'>{counter}</h2>
        </div>
        <div className='flex gap-16 '>
          <button onClick={() => setCounter((prev) => prev + 1)} className='h-12'>Increase</button>
          <button onClick={()=> setCounter((prev) => prev - 1)} className='h-12'>Decrease</button>
          <div className='gap-8'>
            <input onChange={handleCount} placeholder='Enter a number' className='h-12' type="text" />
            <button onClick={() => setCounter((prev) => prev + parseInt(val))} className=''>Add</button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
