import threading
import time
import unittest
from example_fastapi_server.inference_priority import InferencePriorityGate

class Tests(unittest.TestCase):
    def test_all_live_chunks_retained_until_final_ready(self):
        g=InferencePriorityGate();entered=threading.Event();done=threading.Event();received=[]
        def live():
            entered.set()
            for chunk in range(20):
                with g.enter('realtime'):received.append(chunk)
            done.set()
        with g.enter('main'):
            t=threading.Thread(target=live);t.start();self.assertTrue(entered.wait(1));self.assertFalse(done.wait(.03));self.assertEqual(received,[])
        t.join(1);self.assertEqual(received,list(range(20)));self.assertGreaterEqual(g.snapshot()['liveDelays'],1)
    def test_final_overtakes_active_live_and_exceptions_release(self):
        g=InferencePriorityGate();started=threading.Event();finished=threading.Event()
        def final():
            started.set()
            try:
                with g.enter('main'):raise ValueError('test')
            except ValueError:finished.set()
        with g.enter('realtime'):
            t=threading.Thread(target=final);t.start();self.assertTrue(started.wait(1));self.assertTrue(finished.wait(1))
        t.join(1);self.assertTrue(finished.is_set())
        with g.enter('ultrafast'):pass
        self.assertFalse(g.snapshot()['finalActive'])
    def test_main_overtakes_old_live_but_delays_new_live(self):
        g = InferencePriorityGate()
        main_entered = threading.Event()
        release_main = threading.Event()
        live_attempted = threading.Event()
        new_live_entered = threading.Event()

        def main():
            with g.enter('main'):
                main_entered.set()
                release_main.wait(2)

        def new_live():
            live_attempted.set()
            with g.enter('ultrafast'):
                new_live_entered.set()

        with g.enter('realtime'):
            main_thread = threading.Thread(target=main)
            live_thread = threading.Thread(target=new_live)
            main_thread.start()
            try:
                self.assertTrue(main_entered.wait(1))
                self.assertEqual(g.snapshot()['activeLiveOperations'], 1)
                live_thread.start()
                self.assertTrue(live_attempted.wait(1))
                self.assertFalse(new_live_entered.wait(.03))
            finally:
                release_main.set()
                main_thread.join(1)
                if live_thread.ident is not None:
                    live_thread.join(1)
            self.assertTrue(new_live_entered.is_set())
        self.assertEqual(g.snapshot()['activeLiveOperations'], 0)
        self.assertEqual(g.snapshot()['drops'], 0)

    def test_main_requests_still_serialize(self):
        g = InferencePriorityGate()
        attempted = threading.Event()
        entered = threading.Event()

        def second_main():
            attempted.set()
            with g.enter('main'):
                entered.set()

        with g.enter('main'):
            worker = threading.Thread(target=second_main)
            worker.start()
            self.assertTrue(attempted.wait(1))
            self.assertFalse(entered.wait(.03))
        worker.join(1)
        self.assertTrue(entered.is_set())
        self.assertFalse(g.snapshot()['finalActive'])

    def test_two_live_lanes_can_overlap(self):
        g=InferencePriorityGate();done=threading.Event()
        def other():
            with g.enter('ultrafast'):done.set()
        with g.enter('realtime'):
            t=threading.Thread(target=other);t.start();self.assertTrue(done.wait(1))
        t.join(1)

if __name__=='__main__':unittest.main()
