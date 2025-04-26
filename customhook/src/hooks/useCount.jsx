import React, { useState } from 'react'

function useCount() {
    const [count, setCount] = useState(0)
    const increment = (value) => setCount(value)
    return [count, increment]
}

export default useCount
