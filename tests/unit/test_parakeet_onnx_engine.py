import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from RealtimeSTT.transcription_engines.base import TranscriptionEngineConfig, TranscriptionEngineError
from RealtimeSTT.transcription_engines.parakeet_onnx_engine import ParakeetOnnxEngine
from RealtimeSTT.transcription_engines.factory import get_supported_transcription_engines

class Options:
    def __init__(self): self.entries={};self.terminate=False
    def add_session_config_entry(self,k,v):self.entries[k]=v

class Session:
    sessions=[]
    def __init__(self,path,**kw):self.path=path;self.kw=kw;self.calls=[];self.block=False;self.entered=threading.Event();self.sessions.append(self)
    def get_inputs(self):return [NS(name='waveforms'),NS(name='waveforms_lens')]
    def run(self,outputs,inputs,run_options=None):
        self.calls.append(run_options);self.entered.set()
        if self.block:
            deadline=time.monotonic()+2
            while not run_options.terminate and time.monotonic()<deadline:time.sleep(.001)
            if run_options.terminate:raise RuntimeError('terminated')
        return [np.zeros((1,128,5),dtype=np.float32),np.array([5],dtype=np.int64)]

class Model:
    def __init__(self,files,prep,options):
        self.prep=prep('nemo128');self._encoder=Session(str(files['encoder']),**options);self._decoder_joint=Session(str(files['decoder_joint']),**options)
    def recognize_batch(self,w,l):
        self.prep(w,l);self._encoder.run(None,{});self._decoder_joint.run(None,{})
        return [NS(text=' complete German ending ')]

ORT=NS(SessionOptions=Options,RunOptions=Options,InferenceSession=Session,ExecutionMode=NS(ORT_SEQUENTIAL=0),GraphOptimizationLevel=NS(ORT_ENABLE_ALL=99))

class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        for f in ['config.json','vocab.txt','encoder-model.int8.onnx','decoder_joint-model.int8.onnx','nemo128.onnx']:(self.root/f).write_bytes(b'x')
        self.config=TranscriptionEngineConfig(model=str(self.root),device='cpu')
    def tearDown(self):self.tmp.cleanup()
    def make(self):return ParakeetOnnxEngine(self.config,ort_module=ORT,model_cls=Model)
    def test_cpu_only_threads_reference_and_metadata(self):
        e=self.make();r=e.transcribe(np.zeros(16000),language='de')
        self.assertEqual(r.text,'complete German ending');self.assertIsNone(r.info.language)
        self.assertEqual(e.model._encoder.kw['providers'],['CPUExecutionProvider'])
        self.assertEqual(e.model._encoder.kw['sess_options'].intra_op_num_threads,7)
        self.assertEqual(e.model._decoder_joint.kw['sess_options'].intra_op_num_threads,1)
        self.assertEqual(e._preprocessor.kw['sess_options'].intra_op_num_threads,1)
        self.assertTrue(e._preprocessor.path.endswith('nemo128.onnx'))
        self.assertEqual(r.metadata['decoding'],'greedy')
        self.assertIn('parakeet_onnx',get_supported_transcription_engines())
    def test_reject_gpu_missing_and_unknown_options(self):
        self.config.device='cuda'
        with self.assertRaises(TranscriptionEngineError):self.make()
        self.config.device='cpu';self.config.engine_options={'beam':1}
        with self.assertRaises(TranscriptionEngineError):self.make()
        self.config.engine_options={};(self.root/'nemo128.onnx').unlink()
        with self.assertRaises(TranscriptionEngineError):self.make()
    def test_invalid_input_and_closed(self):
        e=self.make()
        for a in [[],[[0,1]],[np.nan]]:
            with self.assertRaises(TranscriptionEngineError):e.transcribe(a)
        e.close()
        with self.assertRaises(TranscriptionEngineError):e.transcribe([0])
    def test_precancel_and_reuse(self):
        e=self.make();cancel=threading.Event();cancel.set()
        with self.assertRaises(TranscriptionEngineError):e.transcribe_cancellable([0],cancel_event=cancel)
        self.assertEqual(e.transcribe([0]).text,'complete German ending')
    def test_active_native_cancel_and_successor(self):
        e=self.make();session=e.model._encoder.session;session.block=True;errors=[]
        def run():
            try:e.transcribe([0])
            except TranscriptionEngineError as exc:errors.append(str(exc))
        t=threading.Thread(target=run);t.start();self.assertTrue(session.entered.wait(1));self.assertTrue(e.cancel_active());t.join(1)
        self.assertFalse(t.is_alive());self.assertTrue(errors);session.block=False
        self.assertEqual(e.transcribe([0]).text,'complete German ending')
        self.assertFalse(e.cancel_active())

if __name__=='__main__':unittest.main()
