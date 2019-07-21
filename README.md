# ssd-extention-study
pytorch版SSDについて以下の改造を行ったもの

* iterationベースからepochベースへの変更
* SSD512(入力サイズ512x512)のサポート
* warmup (burnin)の追加
* focal lossの追加
* prediction moduleの追加
* deconvolutionの追加
* FSSDの追加
