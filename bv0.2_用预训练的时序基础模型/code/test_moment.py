import argparse
import torch
from safetensors.torch import load_file
from model.moment_loader import load_moment
from model.moment_wrapper import segment, stitch, observation_mask, reconstruct
from loss.reconstruction import masked_mse


def check(source):
    torch.set_num_threads(4)
    torch.manual_seed(42)
    models=[load_moment(source,mode) for mode in ('full','head')]
    a,b=models
    state=load_file(str(source/'model.safetensors'))
    for name,p in a.state_dict().items():
        torch.testing.assert_close(p,state[name],rtol=0,atol=0)
        q=b.state_dict()[name]
        assert p.data_ptr()!=q.data_ptr()
        torch.testing.assert_close(p,q,rtol=0,atol=0)
    assert {n for n,p in b.named_parameters() if p.requires_grad}=={n for n,p in b.named_parameters() if n.startswith('head.')}
    for length in (120,1920):
        wave=torch.randn(2,1,length)
        x,v=segment(wave)
        torch.testing.assert_close(stitch(x,2,length),wave[:,0])
        mask=lambda: observation_mask(v,.3,torch.Generator().manual_seed(42))
        mask1=mask()
        assert torch.equal(mask1,mask()) and not (mask1&~v).any()
        assert torch.equal((v&~mask1).sum(-1), (v.sum(-1)/8*.3).round()*8)
        a.eval()
        with torch.no_grad():
            pred=reconstruct(a,x,v,mask1)
            changed=x.masked_fill(~mask1[:,None],999)
            torch.testing.assert_close(pred,reconstruct(a,changed,v,mask1),rtol=0,atol=0)
            torch.testing.assert_close(pred,a(x_enc=changed,input_mask=v.long(),mask=mask1.long()).reconstruction)
            if length==120:
                short=a(x_enc=x[:,:,:120],input_mask=v[:,:120].long(),
                        mask=mask1[:,:120].long()).reconstruction
                torch.testing.assert_close(pred[:,:,:120],short,rtol=1e-4,atol=1e-5)
            pred2=pred.masked_fill(~v[:,None],float('nan'))
            torch.testing.assert_close(masked_mse(pred,x,v,mask1),masked_mse(pred2,x,v,mask1),rtol=0,atol=0)
    # Check the shared path with distinct single-modality gradients.
    parameter=a.head.linear.weight
    individual=[]
    for length in (120,1920):
        x,v=segment(torch.randn(2,1,length))
        m=observation_mask(v,.3,torch.Generator().manual_seed(42))
        loss=masked_mse(reconstruct(a,x,v,m),x,v,m)
        grad,=torch.autograd.grad(loss,parameter)
        assert torch.isfinite(grad).all() and grad.norm()>0
        individual.append(grad)
    assert not torch.equal(*individual)
    other_heads = [m.head.linear.weight.detach().clone() for m in models]
    for index,model in enumerate(models):
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
        for length in (120,512):
            x,v=segment(torch.randn(2,1,length))
            m=observation_mask(v,.3,torch.Generator().manual_seed(12))
            pred=reconstruct(model,x,v,m)
            masked_mse(pred,x,v,m).backward()
            assert model.head.linear.weight.grad.norm()>0
        optimizer.step()
        untouched = models[1-index]
        torch.testing.assert_close(untouched.head.linear.weight,other_heads[1-index],rtol=0,atol=0)
        other_heads[index] = model.head.linear.weight.detach().clone()
        assert not torch.equal(model.head.linear.weight,state['head.linear.weight'])
    # Parameters receiving gradients include the encoder blocks in full mode only.
    assert any(n.startswith('encoder.block') and p.grad is not None and not torch.equal(p,state[n]) for n,p in a.named_parameters())
    assert all(torch.equal(p,state[n]) for n,p in b.named_parameters() if not n.startswith('head.'))
    x,v=segment(torch.randn(2,1,120))
    m=observation_mask(v,.6,torch.Generator().manual_seed(12))
    frozen=load_moment(source,'zero').eval()
    frozen.load_state_dict(a.state_dict(),strict=True)
    torch.testing.assert_close(reconstruct(a,x,v,m),reconstruct(frozen,x,v,m),rtol=0,atol=0)
    print('Official state, independence, native segmentation, masks, padding, loss, optimization and restore passed')


if __name__=='__main__':
    from pathlib import Path
    parser=argparse.ArgumentParser()
    parser.add_argument('--official',type=Path,required=True)
    check(parser.parse_args().official)
