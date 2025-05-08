import React from 'react';

const TaskCard = ({ task, deleteTask }) => {
    return (
        <div className="flex items-center justify-between w-full px-5 py-4 bg-white border-l-4 border-blue-500 rounded-xl shadow-md hover:shadow-lg transition duration-300">
            <p className="text-gray-800 font-medium text-base break-words">{task}</p>
            <span className="text-green-500 text-lg">✅</span>
            <button onClick={deleteTask} className='p-2 rounded bg-red-500 text-white'>Delete</button>
        </div>
    );
};

export default TaskCard;
