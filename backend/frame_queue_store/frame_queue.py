import asyncio


class FrameQueue:
    def __init__(self):
        self.queue = asyncio.Queue()

    async def put(self, item):
        await self.queue.put(item)

    async def get(self):
        return await self.queue.get()

    def task_done(self):
        self.queue.task_done()

    def empty(self):
        return self.queue.empty()

    def size(self):
        return self.queue.qsize()


frame_queue = FrameQueue()
