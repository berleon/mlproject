import os


def get_tensorboard_dir(model_prefix, tensorboard_dir=None):
    if tensorboard_dir is None:
        if 'TENSORBOARD_DIR' in os.environ:
            tensorboard_dir = os.environ['TENSORBOARD_DIR']
        else:
            raise ValueError("No path for tensorboard_dir given and "
                             "TENSORBOARD_DIR enviroment variable is also not set.")
    return os.path.join(tensorboard_dir, model_prefix)


class DevNullSummaryWriter:
    def add_scalar(self, tag, scalar_value, global_step=None):
        pass

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None):
        pass

    def export_scalars_to_json(self, path):
        pass

    def add_histogram(self, tag, values, global_step=None, bins='tensorflow'):
        pass

    def add_image(self, tag, img_tensor, global_step=None):
        pass

    def add_video(self, tag, vid_tensor, global_step=None):
        pass

    def add_audio(self, tag, snd_tensor, global_step=None, sample_rate=44100):
        pass

    def add_text(self, tag, text_string, global_step=None):
        pass

    def add_graph_onnx(self, prototxt):
        pass

    def add_graph(self, model, input_to_model, verbose=False):
        pass

    def add_embedding(self, mat, metadata=None, label_img=None, global_step=None, tag='default',
                      metadata_header=None):
        pass

    def add_pr_curve(self, tag, labels, predictions, global_step=None, num_thresholds=127,
                     weights=None):
        pass

    def add_pr_curve_raw(self, tag, true_positive_counts,
                         false_positive_counts,
                         true_negative_counts,
                         false_negative_counts,
                         precision,
                         recall, global_step=None, num_thresholds=127, weights=None):
        pass

    def close(self):
        pass


_global_writer = DevNullSummaryWriter
_global_step = None


def set_global_writer(writer):
    global _global_writer
    _global_writer = writer


def set_global_step(global_step):
    global _global_step
    _global_step = global_step


def add_scalar(tag, scalar_value, global_step=None):
    global_step = global_step or _global_step
    _global_writer.add_scalar(tag, scalar_value, global_step)


def add_scalars(main_tag, tag_scalar_dict, global_step=None):
    global_step = global_step or _global_step
    _global_writer.add_scalars(main_tag, tag_scalar_dict, global_step)


def add_histogram(tag, values, global_step=None, bins='tensorflow'):
    global_step = global_step or _global_step
    _global_writer.add_histogram(tag, values, global_step, bins)


def add_image(tag, img_tensor, global_step=None):
    global_step = global_step or _global_step
    _global_writer.add_image(tag, img_tensor, global_step)


def add_video(tag, vid_tensor, global_step=None):
    global_step = global_step or _global_step
    _global_writer.add_video(tag, vid_tensor, global_step=None)


def add_audio(tag, snd_tensor, global_step=None, sample_rate=44100):
    global_step = global_step or _global_step
    _global_writer.add_audio(tag, snd_tensor, global_step, sample_rate)


def add_text(tag, text_string, global_step=None):
    global_step = global_step or _global_step
    _global_writer.add_text(tag, text_string, global_step)


def add_graph_onnx(prototxt):
    _global_writer.add_graph_onnx(prototxt)


def add_graph(model, input_to_model, verbose=False):
    _global_writer.add_graph(model, input_to_model, verbose)


def add_embedding(mat, metadata=None, label_img=None, global_step=None, tag='default',
                  metadata_header=None):
    global_step = global_step or _global_step
    _global_writer.add_embedding(mat, metadata, label_img, global_step, tag, metadata_header)


def add_pr_curve(tag, labels, predictions, global_step=None, num_thresholds=127,
                 weights=None):
    global_step = global_step or _global_step
    _global_writer.add_pr_curve(tag, labels, predictions, global_step, num_thresholds, weights)


def add_pr_curve_raw(tag, true_positive_counts,
                     false_positive_counts,
                     true_negative_counts,
                     false_negative_counts,
                     precision,
                     recall, global_step=None, num_thresholds=127, weights=None):
    global_step = global_step or _global_step
    _global_writer.add_pr_curve_raw(
        tag, true_positive_counts, false_positive_counts, true_negative_counts,
        false_negative_counts, precision, recall, global_step, num_thresholds, weights)

