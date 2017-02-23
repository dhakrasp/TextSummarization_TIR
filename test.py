from helper import *
from keras.models import model_from_json

def test(txt, src_tokenizer, target_vocab, model, batch_size):
    X, _ = preprocess_data(txt, src_tokenizer, tar_text=None, tar_tokenizer=None)
    pred = model.predict_classes(X, batch_size=batch_size)

    outputs = decode(pred, target_vocab)
    print('\n\n\n')
    print('-' * 50 + '\t\tTesting\t\t' + '-' * 50)
    print(outputs)


if __name__ == '__main__':
    model_file = 'model'
    weights_file = 'weights'
    src_file_name = "source/50"
    batch_size = 20
    test_txt = []
    for i in range(1):
        test_txt.append(get_file_content(src_file_name))
    model = model_from_json(model_file)
    model.compile()
    test(test_txt, src_tokenizer, tar_tokenizer.vocab, model, batch_size)
