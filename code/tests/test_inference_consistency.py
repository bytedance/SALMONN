"""Train/inference parity for RVQ precision, style features and cached targets."""
import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from dataset import Gesture_Dataset
from models.vq.model import RVQVAE
from models.vq.precision import configure_encoding_precision,resolve_encoding_precision
from scripts.expression_io import load_flame_style_features,resolve_style_flame_path
from scripts import infer_ellsa2_gesture_expression_from_audio as inference

class FixedEncoder(nn.Module):
    def __init__(self, value):
        super().__init__();self.anchor=nn.Parameter(torch.zeros(1));self.value=value;self.seen=None
    def encode(self,x):
        self.seen=x.detach().cpu().clone()
        return torch.full((len(x),x.shape[1]//4,6),self.value,dtype=torch.long,device=x.device),None

class EncodingConsistencyTests(unittest.TestCase):
    def tiny_model(self):
        torch.manual_seed(7)
        args=SimpleNamespace(causal=True,num_quantizers=2,shared_codebook=False,quantize_dropout_prob=0,mu=.99)
        model=RVQVAE(args,input_width=3,nb_code=16,code_dim=8,output_emb_width=8,down_t=2,stride_t=2,width=8,depth=2,dilation_growth_rate=2).eval().set_context_frames(12)
        for layer in model.quantizer.layers:layer.codebook.normal_()
        return model

    def test_precision_follows_checkpoint_and_explicit_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'config.json').write_text(json.dumps({'rvq_encoder_dtype':'float64','rvq_quantizer_distance_dtype':'float64'}))
            args=SimpleNamespace(model_path=str(root),rvq_encoder_dtype=None)
            self.assertEqual(resolve_encoding_precision(args,'rvq_encoder_dtype'),'float64')
            args.rvq_encoder_dtype='float32'
            self.assertEqual(resolve_encoding_precision(args,'rvq_encoder_dtype'),'float32')
            (root/'config.json').write_text(json.dumps({'rvq_cache_policy':{'encoder_dtype':'float64'}}));args.rvq_encoder_dtype=None
            self.assertEqual(resolve_encoding_precision(args,'rvq_encoder_dtype'),'float64')
            self.assertEqual(resolve_encoding_precision(args,'rvq_quantizer_distance_dtype'),'float32')

    def test_encoder_precision_does_not_change_decoder_or_weights(self):
        model=self.tiny_model();codes=torch.randint(0,16,(1,9,2))
        with torch.no_grad():expected=model.forward_decoder(codes).clone()
        state={k:v.clone() for k,v in model.state_dict().items()}
        configure_encoding_precision(model,'float64','float64')
        self.assertEqual(next(model.encoder.parameters()).dtype,torch.float64)
        self.assertEqual(next(model.decoder.parameters()).dtype,torch.float32)
        with torch.no_grad():
            actual=model.forward_decoder(codes);encoded,_=model.encode(torch.randn(1,24,3))
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        self.assertEqual(encoded.shape,(1,6,2))
        for k,v in model.state_dict().items():torch.testing.assert_close(v.float(),state[k].float(),rtol=0,atol=0)
        configure_encoding_precision(model,'float32','float32')
        self.assertEqual(next(model.encoder.parameters()).dtype,torch.float32)

    def test_double_distance_resolves_float32_cancellation(self):
        model=self.tiny_model();layer=model.quantizer.layers[0]
        layer.codebook.fill_(-1000);layer.codebook[0].zero_();layer.codebook[1].zero_()
        layer.codebook[:2,0]=100000;layer.codebook[1,1]=1
        x=torch.zeros(1,8);x[0,0]=100000;x[0,1]=.9
        configure_encoding_precision(model,'float64','float64')
        self.assertEqual(int(layer.quantize(x).item()),1)

    def test_training_cache_key_changes_with_precision_and_context(self):
        ds=Gesture_Dataset.__new__(Gesture_Dataset);ds.has_gesture_target=True;ds.has_expression_target=False;ds.data=[{'pose_file':'same.npz'}]
        ds.args=SimpleNamespace(rvq_encoder_dtype='float32',rvq_quantizer_distance_dtype='float32',rvq_gesture_context_frames=300,rvq_expression_context_frames=240)
        first=ds.motion_cache_sample_key(0);ds.args.rvq_encoder_dtype='float64'
        self.assertNotEqual(first,ds.motion_cache_sample_key(0));second=ds.motion_cache_sample_key(0);ds.args.rvq_gesture_context_frames=0
        self.assertNotEqual(second,ds.motion_cache_sample_key(0))

class StyleConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        rng=np.random.default_rng(4);self.raw=rng.normal(size=(32,112)).astype(np.float32)
        self.flame=self.root/'style_flame.npz';np.savez(self.flame,motioncode=self.raw,valid_mask=np.ones(32,bool),motion_fps=30)
        self.pose=self.root/'style.npz';np.savez(self.pose,body_pose=np.zeros((32,63),np.float32),left_hand_pose=np.zeros((32,45),np.float32),right_hand_pose=np.zeros((32,45),np.float32),mocap_frame_rate=30)
        self.mean=rng.normal(size=108).astype(np.float32);self.std=rng.uniform(.5,2,size=108).astype(np.float32)
        self.stats={k:torch.zeros(n) if k.startswith('mean') else torch.ones(n) for k,n in [('mean_upper',78),('std_upper',78),('mean_left_hand',90),('std_left_hand',90),('mean_right_hand',90),('std_right_hand',90),('mean_transv',3),('std_transv',3)]}
        self.args=SimpleNamespace(interleave_motion_parts=False,motion_tokenizer_type='upper_hands',gesture_num_codebooks=12,expression_num_codebooks=6,num_motion_codebooks=18,motion_style_expression=True,flame_fps=30)

    def test_style_features_exactly_match_training_dataset(self):
        ds=Gesture_Dataset.__new__(Gesture_Dataset)
        motion=ds._select_unils_motioncode(self.raw).astype(np.float32)
        ds._load_flame_record=lambda path:{'flame:motioncode':motion}
        ds._get_expression_slice_indices=lambda start,end:(start,end)
        ds.motion_mean=self.mean;ds.motion_std=self.std
        expected=ds._build_expression_motion_feature_parts(str(self.flame),4,28)['flame_motioncode']
        actual=load_flame_style_features(str(self.flame),4,24,self.mean,self.std)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    def test_full_style_has_expression_while_pose_guide_stays_masked(self):
        decoders={'upper':FixedEncoder(3),'hands':FixedEncoder(7)};face=FixedEncoder(11)
        codes=inference.encode_motion_style_tokens(str(self.pose),str(self.flame),0,24,decoders,self.stats,face,self.mean,self.std,self.args)
        self.assertTrue((codes[:,:6]==3).all());self.assertTrue((codes[:,6:12]==7).all());self.assertTrue((codes[:,12:]==11).all())
        guide=inference.encode_pose_tokens(str(self.pose),0,24,decoders,self.stats,self.args)
        self.assertTrue((guide[:,12:]==-100).all())
        self.args.motion_style_expression=False
        legacy=inference.encode_motion_style_tokens(str(self.pose),'missing.npz',0,24,decoders,self.stats,face,self.mean,self.std,self.args)
        torch.testing.assert_close(legacy,guide,rtol=0,atol=0)

    def test_missing_invalid_and_wrong_fps_style_fail(self):
        with self.assertRaises(FileNotFoundError):resolve_style_flame_path(str(self.pose),'missing.npz')
        np.savez(self.flame,motioncode=self.raw,valid_mask=np.zeros(32,bool),motion_fps=30)
        with self.assertRaisesRegex(ValueError,'invalid frames'):load_flame_style_features(str(self.flame),0,24,self.mean,self.std)
        np.savez(self.flame,motioncode=self.raw,valid_mask=np.ones(8,bool),motion_fps=30)
        with self.assertRaisesRegex(ValueError,'invalid frames'):load_flame_style_features(str(self.flame),0,24,self.mean,self.std)
        np.savez(self.flame,motioncode=self.raw,valid_mask=np.ones(32,bool),motion_fps=25)
        with self.assertRaisesRegex(ValueError,'frame rate'):load_flame_style_features(str(self.flame),0,24,self.mean,self.std)

    def test_prepare_testset_carries_flame_reference(self):
        from scripts.prepare_testset import prepare
        path=prepare(self.root/'jobs',test_id='fro_00038');job=json.loads(path.read_text())['data'][0]
        self.assertTrue(Path(job['motion_style_flame_path']).is_file())
        with np.load(job['motion_style_flame_path']) as data:
            self.assertEqual(data['motioncode'].shape,(612,108));self.assertTrue(data['valid_mask'].all())

class ModelInputConsistencyTests(unittest.TestCase):
    def test_inference_prompt_is_exact_training_prefix(self):
        from dataset import build_alternating_qwen_sequence
        slot='<|vision_pad|>'
        for history in [0,2,14]:
            for dialog in [False,True]:
                for speaker in ['', 'P0001']:
                    guides=[0]*(history+1);guides[0]=6
                    training=build_alternating_qwen_sequence(history+1,slot,dialog,False,6,18,
                        speaker_id=speaker,generation_target='gesture_expression',prefix_motion_steps=150,
                        inline_exemplar_steps_by_window=guides)
                    suffix=slot*6+'<|im_end|>\n'
                    self.assertTrue(training.endswith(suffix))
                    actual=inference.build_streaming_gesture_expression_sequence(history,slot,dialog,6,
                        speaker_id=speaker,prefix_motion_steps=150,inline_exemplar_steps_by_window=guides)
                    self.assertEqual(actual,training[:-len(suffix)])

    def test_fbank_pipeline_matches_training_for_same_available_audio(self):
        from dataset import CoSpeechGestureCollator
        extractor=inference.Fbank(inference.FbankConfig(num_mel_bins=128))
        collator=CoSpeechGestureCollator.__new__(CoSpeechGestureCollator)
        collator.fbank=extractor;collator.audio_sr=16000
        t=torch.arange(38400)/16000
        audio=[torch.sin(2*torch.pi*220*t)*.1,torch.zeros(12800)]
        expected,lengths=collator._build_fbank_batch(audio)
        actual,actual_lengths=inference.build_fbank_batch(extractor,audio,16000)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        torch.testing.assert_close(actual_lengths,lengths,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
