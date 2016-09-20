**This implementation is currently still under construction.**

## WaveNet: A Generative Model for Raw Audio

This is the Chainer implementation of [WaveNet](http://arxiv.org/abs/1609.03499)

この[記事](http://musyoku.github.io/2016/09/18/wavenet-a-generative-model-for-raw-audio/)で実装したコードです。

まだ完成していませんが音声の生成はできます。

#### Todo:
- [ ] Local conditioning
- [ ] Global conditioning
- [ ] Training on CSTR VCTK Corpus

## Running

### Requirements

- Chainer 1.12
- scipy.io.wavfile

## 図とか

![input](https://raw.githubusercontent.com/musyoku/musyoku.github.io/562b128139d4d52f3105c17a366934f92ff82613/images/post/2016-09-17/wavenet_input.png)

![dilated conv](http://i.imgur.com/hTWShQF.gif)