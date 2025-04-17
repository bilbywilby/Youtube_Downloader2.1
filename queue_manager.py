from queue import Queue
from threading import Thread
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Task:
    """Represents a task to be processed."""
    def __init__(self, name, data=None):
        self.name = name
        self.data = data

    def __str__(self):
        return f"Task(name='{self.name}')"

    def process(self):
        """Simulates the processing of the task."""
        logging.info("Processing task: %s", self)
        time.sleep(2)
        logging.info("Task completed: %s", self)

class TaskQueue:
    """Manages a queue of tasks and their processing."""
    def __init__(self):
        self._queue = Queue()
        self._processor_thread = None
        self._is_processing = False

    def enqueue_task(self, task: Task):
        """Adds a task to the queue."""
        self._queue.put(task)
        logging.info(f"Task enqueued: {task}")

    def dequeue_task(self) -> Task | None:
        """Removes and returns the next task from the queue."""
        if not self._queue.empty():
            task = self._queue.get()
            logging.info(f"Task dequeued: {task}")
            return task
        logging.info("Queue is empty, cannot dequeue.")
        return None

    def _process_tasks(self):
        """Internal method to process tasks from the queue."""
        if self._is_processing:
            logging.warning("Task processing already in progress.")
            return

        self._is_processing = True
        logging.info("Starting task processing.")
        while not self._queue.empty():
            task = self._queue.get()
            task.process()
            self._queue.task_done()  # Indicate that the task is complete
        self._is_processing = False
        logging.info("All tasks processed.")

    def start_processor(self):
        """Starts the task processor in a separate thread."""
        if self._processor_thread is None or not self._processor_thread.is_alive():
            self._processor_thread = Thread(target=self._process_tasks, daemon=True)
            self._processor_thread.start()
            logging.info("Task processor started in a new thread.")
        else:
            logging.info("Task processor is already running.")

    def join(self, timeout=None):
        """Blocks until all tasks in the queue have been processed."""
        self._queue.join()
        if self._processor_thread and self._processor_thread.is_alive():
            self._processor_thread.join(timeout=timeout)
            logging.info("Task processor thread finished.")
        else:
            logging.info("No active task processor thread to join.")

# Example usage
if __name__ == "__main__":
    task_queue = TaskQueue()
    task_queue.enqueue_task(Task("Download video 1"))
    task_queue.enqueue_task(Task("Process video 1"))
    task_queue.enqueue_task(Task("Download video 2"))
    task_queue.enqueue_task(Task("Process video 2"))

    task_queue.start_processor()

    # Optionally add more tasks later
    time.sleep(5)
    task_queue.enqueue_task(Task("Upload results"))

    # Wait for all tasks to complete
    task_queue.join()

    logging.info("All operations completed.")
# This code provides a simple task queue manager that allows for adding tasks,
# processing them in a separate thread, and waiting for all tasks to complete.
# It uses Python's built-in Queue and Thread classes to manage concurrency.
# The logging module is used to provide detailed information about the state
# of the queue and the tasks being processed.