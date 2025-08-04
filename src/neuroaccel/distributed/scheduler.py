"""Distributed processing framework for GPU computations."""
from typing import Optional, List, Dict, Any, Callable
import os
import logging
import asyncio
import json
from datetime import datetime
import threading
import queue
import uuid

import ray
import torch
import numpy as np

from neuroaccel.core.gpu import GPUManager, DeviceType

logger = logging.getLogger(__name__)

@ray.remote(num_gpus=1)
class GPUWorker:
    """Worker class for GPU computations."""

    def __init__(self, device_index: int = 0):
        """Initialize GPU worker.

        Args:
            device_index: GPU device index
        """
        self.device_index = device_index
        self.gpu_manager = GPUManager(
            device="cuda",
            device_index=device_index
        )
        self.device = self.gpu_manager.get_device()
        self.last_heartbeat = datetime.now()
        self.current_task = None
        self.error_count = 0

    def health_check(self) -> Dict[str, Any]:
        """Check worker health status.

        Returns:
            Dict containing health metrics
        """
        memory_info = self.gpu_manager.memory_info()
        device_status = {
            'is_alive': True,
            'last_heartbeat': self.last_heartbeat,
            'error_count': self.error_count,
            'current_task': self.current_task,
            **memory_info
        }
        self.last_heartbeat = datetime.now()
        return device_status

    def process(
        self,
        function: Callable,
        data: torch.Tensor,
        task_id: str = None,
        **kwargs
    ) -> torch.Tensor:
        """Process data using provided function.

        Args:
            function: Processing function
            data: Input data tensor
            task_id: ID of current task
            **kwargs: Additional arguments for function

        Returns:
            Processed data tensor
        """
        try:
            self.current_task = task_id
            data = data.to(self.device)
            result = function(data, **kwargs)
            return result.cpu()
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error in worker {self.device_index} processing task {task_id}: {e}")
            raise
        finally:
            self.current_task = None

class TaskScheduler:
    """Scheduler for distributing tasks across GPU workers."""

    def __init__(
        self,
        num_gpus: Optional[int] = None,
        memory_fraction: float = 0.9,
        health_check_interval: float = 30.0  # seconds
    ):
        """Initialize task scheduler.

        Args:
            num_gpus: Number of GPUs to use (None for all available)
            memory_fraction: Fraction of GPU memory to use
            health_check_interval: Interval between worker health checks
        """
        # Initialize Ray if not already running
        if not ray.is_initialized():
            ray.init()

        self.num_gpus = num_gpus or torch.cuda.device_count()
        self.memory_fraction = memory_fraction
        self.health_check_interval = health_check_interval
        self.last_health_check = datetime.now()

        # Create worker pool
        self.workers = [
            GPUWorker.remote(i) for i in range(self.num_gpus)
        ]

        # Use PriorityQueue for task scheduling
        self.task_queue = queue.PriorityQueue()
        self.result_queue = queue.Queue()

        # Task tracking
        self.tasks: Dict[str, Dict] = {}
        self.worker_tasks: Dict[int, str] = {}

        # Performance metrics
        self.metrics = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_processing_time': 0.0,
            'avg_processing_time': 0.0
        }

        # Start scheduler thread
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self.scheduler_thread.start()

        logger.info(f"Task scheduler initialized with {self.num_gpus} GPUs")

    def submit_task(
        self,
        function: Callable,
        data: torch.Tensor,
        metadata: Optional[Dict] = None,
        priority: int = 0,
        max_retries: int = 3,
        **kwargs
    ) -> str:
        """Submit task for processing.

        Args:
            function: Processing function
            data: Input data tensor
            metadata: Optional task metadata
            priority: Task priority (higher value = higher priority)
            max_retries: Maximum number of retry attempts
            **kwargs: Additional arguments for function

        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())
        task = {
            'id': task_id,
            'function': function,
            'data': data,
            'kwargs': kwargs,
            'metadata': metadata or {},
            'status': 'pending',
            'submit_time': datetime.now(),
            'start_time': None,
            'end_time': None,
            'result': None,
            'error': None,
            'priority': priority,
            'retry_count': 0,
            'max_retries': max_retries
        }

        self.tasks[task_id] = task

        # Use PriorityQueue to handle task priorities
        self.task_queue.put((-priority, task_id))  # Negative for higher priority first

        logger.info(f"Task {task_id} submitted with priority {priority}")
        return task_id

    def get_task_status(self, task_id: str) -> Dict:
        """Get status of a task.

        Args:
            task_id: Task ID

        Returns:
            Task status information
        """
        if task_id not in self.tasks:
            raise ValueError(f"Unknown task ID: {task_id}")

        task = self.tasks[task_id]
        return {
            'id': task['id'],
            'status': task['status'],
            'metadata': task['metadata'],
            'submit_time': task['submit_time'],
            'start_time': task['start_time'],
            'end_time': task['end_time'],
            'error': task['error']
        }

    def get_result(self, task_id: str, timeout: Optional[float] = None) -> torch.Tensor:
        """Get task result.

        Args:
            task_id: Task ID
            timeout: Optional timeout in seconds

        Returns:
            Processed data tensor
        """
        if task_id not in self.tasks:
            raise ValueError(f"Unknown task ID: {task_id}")

        task = self.tasks[task_id]

        if task['status'] == 'completed':
            return task['result']
        elif task['status'] == 'failed':
            raise RuntimeError(f"Task failed: {task['error']}")

        # Wait for result
        try:
            while True:
                if task['status'] == 'completed':
                    return task['result']
                elif task['status'] == 'failed':
                    raise RuntimeError(f"Task failed: {task['error']}")

                if timeout is not None:
                    if (datetime.now() - task['submit_time']).total_seconds() > timeout:
                        raise TimeoutError("Task timeout")

                asyncio.sleep(0.1)
        except Exception as e:
            task['status'] = 'failed'
            task['error'] = str(e)
            raise

    def _check_worker_health(self):
        """Check health status of all workers."""
        for worker_id, worker in enumerate(self.workers):
            try:
                status = ray.get(worker.health_check.remote())
                if not status['is_alive']:
                    logger.warning(f"Worker {worker_id} appears to be unhealthy")
                    if worker_id in self.worker_tasks:
                        task_id = self.worker_tasks[worker_id]
                        self._handle_failed_task(task_id)
                        del self.worker_tasks[worker_id]
            except Exception as e:
                logger.error(f"Error checking worker {worker_id} health: {e}")

    def _handle_failed_task(self, task_id: str) -> None:
        """Handle a failed task, implementing retry logic.

        Args:
            task_id: ID of failed task
        """
        task = self.tasks[task_id]
        task['retry_count'] += 1

        if task['retry_count'] < task['max_retries']:
            logger.info(f"Retrying task {task_id} (attempt {task['retry_count']})")
            task['status'] = 'pending'
            task['error'] = None
            # Re-queue with original priority
            self.task_queue.put((-task['priority'], task_id))
        else:
            logger.error(f"Task {task_id} failed after {task['retry_count']} attempts")
            task['status'] = 'failed'
            task['end_time'] = datetime.now()

    def _check_completed_tasks(self):
        """Check and update completed tasks."""
        # First check worker health
        self._check_worker_health()

        # Then check task completion
        for worker_id, task_id in self.worker_tasks.items():
            task = self.tasks[task_id]
            worker = self.workers[worker_id]

            if ray.get(worker.ready.remote()):
                try:
                    result = ray.get(worker.get_result.remote())
                    task['result'] = result
                    task['status'] = 'completed'
                    task['end_time'] = datetime.now()
                    logger.info(f"Task {task_id} completed successfully")
                except Exception as e:
                    logger.error(f"Task {task_id} failed: {e}")
                    self._handle_failed_task(task_id)

                del self.worker_tasks[worker_id]

    def _assign_new_tasks(self):
        """Assign pending tasks to available workers."""
        if self.task_queue.empty():
            return

        for worker_id, worker in enumerate(self.workers):
            if worker_id not in self.worker_tasks:
                try:
                    # Get highest priority task
                    _, task_id = self.task_queue.get_nowait()
                except queue.Empty:
                    break

                task = self.tasks[task_id]

                # Skip if task already completed or permanently failed
                if task['status'] in ['completed', 'failed']:
                    continue

                task['status'] = 'running'
                task['start_time'] = datetime.now()

                # Submit task to worker with task ID for tracking
                try:
                    ray.get(worker.process.remote(
                        task['function'],
                        task['data'],
                        task_id=task_id,
                        **task['kwargs']
                    ))
                    self.worker_tasks[worker_id] = task_id
                    logger.info(f"Assigned task {task_id} to worker {worker_id}")
                except Exception as e:
                    logger.error(f"Error assigning task {task_id} to worker {worker_id}: {e}")
                    self._handle_failed_task(task_id)

    def _scheduler_loop(self):
        """Main scheduler loop."""
        while self.running:
            try:
                self._check_completed_tasks()
                self._assign_new_tasks()
                asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                continue

    def shutdown(self):
        """Shutdown scheduler."""
        self.running = False
        self.scheduler_thread.join()
        ray.shutdown()
