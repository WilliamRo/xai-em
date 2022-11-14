from tframe import console
from tframe import mu



def get_initial_model():
  from td_core import th

  model = mu.Predictor(th.mark)
  model.add(mu.Input(sample_shape=th.input_shape))
  return model


def finalize(model):
  from td_core import th
  assert isinstance(model, mu.Predictor)

  # Add last layer
  model.add(mu.HyperConv2D(filters=1, kernel_size=1))

  # Build model
  model.build(loss=th.loss_string, metric=th.loss_string)
  return model


def get_unet(arc_string='8-3-4-2-relu-mp', **kwargs):
  model = get_initial_model()

  mu.UNet(2, arc_string=arc_string, **kwargs).add_to(model)

  return finalize(model)



if __name__ == '__main__':
  # ++ Blue box for model
  from td_core import th

  console.suppress_logging()

  th.input_shape = [64, 64, 1]
  th.mark = 'BB_in_td_mu'
  th.save_model = False

  model = get_unet()

  model.rehearse(export_graph=False, build_model=False, mark=th.mark)
