# 利用手引き：訂正後の応答を調べる・二視点分解の利点を確かめる

[English](GRAIN_EVENT_AND_VIEW_INCREMENT.md) | [简体中文](GRAIN_EVENT_AND_VIEW_INCREMENT_ZH.md) | **日本語**

二つの入口は独立しており、従来の因子得点も参照集団も変更しない。現在の main を取得し、ルートディレクトリで実行する。基準環境は Python 3.12。依存関係は `python -m pip install -r requirements-lock-main.txt` で導入する。以下の例に外部モデルや API は不要である。

## 訂正後、話し手は新しい前提を採用したか

```sh
python -m grain_core.event_response examples/event_response.json
```

[入力例](../examples/event_response.json)では相手が red を blue に訂正し、本人は “I'll use blue.” と応じる。`events[0].status` の `support` は、その場で採用を表明したという判定であり、実行の確認ではない。対象者、二つの前提、訂正のターン、観測の最終ターン（`cutoff`）を指定する。番号は 0 から始まり、最終ターンを含む。対象は訂正直後の本人の応答である。明示的な反証は `refute`、判断保留は `insufficient`。答えが曖昧だった場合と、まだ応答が観測されていない場合は `opportunity` と `reason` で区別する。

組み込み規則は明示的な英語の言い回しに限られる。別の言語や言い換えには、観測済みの発話列に紐付けた外部コーディングを受け付ける。版と指紋を記録し、単なる意味上の候補は判定票に加えない。`--liquid` の軌跡は同一入力の読み取り結果、`later_checks` は後に観察された行動であり、いずれも当初の判断を書き換えない。出力には発話の引用を含むため、非公開の会話では出力の共有先に注意する。

条件別の観測機会 n 件で支持 s 件、反証 f 件なら、未確定の応答を含む割合の範囲は [s/n, 1−f/n]、未観測の条件は [0,1] である。例では観測条件に 0.75、未観測条件に 0.25 を割り当てるため、`reference_rate_bounds` は [0.75, 1.0] となる。これは信頼区間でも特性の保有確率でもない。同じ行動は重複して数えない。

## 二つの視点に共通の履歴を課すと判断は変わるか

```sh
python -m grain_core.dual_view_increment examples/dual_view_increment.json
```

同じ座標・同じ時点で観測した二系列を使う。日付列を付ける場合は同一の ISO 日付などを使う。欠測区間は入力前に分割する（取引日暦の確認は利用者側で行う）。校正を与えなければ二つの水準・平均・符号付き差のみを返す。交差モーメントによる校正では、独立の校正期間の対を `previous_levels` に渡す。分散の推定値が正でなければ `unsupported_calibration` となる。

[数値例](../examples/dual_view_increment.json)の `rates` と `budgets` の順は共有、第1視点、第2視点。各時点の半幅、外側の境界、校正 ID も指定する。当日のみの `static`、共有遷移のみの `common_only`、二系列を別々に伝播させた `simple_lag`、同じ過去状態を要求する `full` を比較する。例では後二者の範囲が [−0.4, 2.0] と [0.8, 1.2] であり、ゼロを超えるかどうかの判断が変わる。

| `vs_simple_lag` | 読み方 |
|---|---|
| `simple_lag_sufficient` | 単純な差分法で同じ範囲が得られる。 |
| `narrows_only` | 範囲は狭くなるが、事前に決めた閾値に関する結論は変わらない。 |
| `changes_threshold_decision` | 共通の履歴を課して初めて閾値を除外できる。 |
| `null` | `status` で校正なし、校正不適合、制約の矛盾などを区別する。ゼロで代用しない。 |

共有成分の符号が決まることと、翌日の寄り付きの方向を当てることは別の問いである。予測用途なら、同じ締切・同じ観測日・同じ履歴を使う基準法と対にして評価する。
