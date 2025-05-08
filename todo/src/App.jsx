import React, { useState } from 'react';
import TaskCard from './components/TaskCard';

const App = () => {
  const [task, setTask] = useState('');
  const [taskList, setTaskList] = useState([]);

  const AddTask = () => {
    if (task.trim()) {
      setTaskList([...taskList, task])
    }
  }

  const deleteTask = (index) => {
    const updatedTaskList = [...taskList];
    updatedTaskList.splice(index, 1);
    setTaskList(updatedTaskList);
  };

  return (
    <div className='bg-[#f7f2ef] flex justify-center items-center w-full h-screen'>
      <div className='relative flex w-[60vw] shadow-lg h-[75vh] rounded-xl overflow-hidden flex-col'>
        <div className='bg-gradient-to-r from-[#614285] to-[#615693] flex justify-center w-full h-60'>
          <h1 className='text-white my-3 text-5xl font-light'>TODO List</h1>
        </div>

        <div className='bg-[#f7f7ff] w-full h-full'></div>

        <div className='inner-container flex flex-col gap-8 absolute left-1/2 -translate-x-1/2 w-[90%] h-[80%] mt-16'>
          <div className='bg-white w-full flex flex-col gap-4 h-[30%] px-32 rounded-xl shadow-lg'>
            <input
              type='text'
              value={task}
              onChange={(e) => setTask(e.target.value)}
              placeholder='Bol aaj kya krega'
              className='w-full mt-4 focus:outline-none focus:border-b border-b py-2'
            />
            <button onClick={AddTask} className="bg-red-400 w-fit mx-auto text-white px-6 py-2 rounded-lg shadow-[0_4px_20px_rgba(255,87,87,0.4)]">
              Add
            </button>


          </div>


          <div className='bg-white w-full h-[70%] overflow-y-auto rounded-xl shadow-lg p-4 flex flex-col gap-4'>

            {taskList.length > 0 ? (
              taskList.map((task, index) => (
                <TaskCard key={index} task={task} deleteTask={() => deleteTask(index)} />
              ))
            ) : (
              <p className='text-gray-400 text-center mt-4'>kuch likh na yrr</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;
