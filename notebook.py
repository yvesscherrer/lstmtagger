import model

# Training
options = model.parser.parse_args([
	"--log-dir", "exp1",
	"--training-data", "uk_iu-ud-train.conllu",
	"--dev-data", "uk_iu-ud-dev.conllu",
	"--num-epochs", "20",
	"--vocab-save", "exp1/vocab.pkl",
	"--settings-save", "exp1/settings.pkl",
	"--params-save", "exp1/params.bin",
	"--debug"
])
model.main(options)

# Testing
options = model.parser.parse_args([
	"--log-dir", "exp1",
	"--vocab", "exp1/vocab.pkl",
	"--settings", "exp1/settings.pkl",
	"--params", "exp1/params.bin",
	"--test-data", "uk_iu-ud-test.conllu",
	"--test-data-out", "testout.txt"
])
model.main(options)
