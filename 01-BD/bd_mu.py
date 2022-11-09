from tframe import console
from tframe import mu



def get_initial_model():
  from bd_core import th

  model = mu.Predictor(th.mark)
  model.add(mu.Input(sample_shape=th.input_shape))
  return model


def finalize(model):
  from bd_core import th
  assert isinstance(model, mu.Predictor)

  # Build model
  model.build(loss=th.loss_string, metric=th.loss_string)
  return model


def get_unet():
  model = get_initial_model()

  mu.UNet(3, arc_string='8-3-4-2-relu-mp').add_to(model)

  return finalize(model)



if __name__ == '__main__':
  # ++ Blue box for model
  from bd_core import th

  console.suppress_logging()

  th.input_shape = [64, 64, 64, 1]
  th.mark = 'BB_in_bd_mu'
  th.save_model = False

  model = get_unet()

  model.rehearse(export_graph=False, build_model=False,
                 path=model.agent.ckpt_dir, mark=th.mark)
