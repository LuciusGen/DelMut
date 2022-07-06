import java.io.File

class clasifierPipeline(
        errorMessage: String,
        command: String
) {
    fun run(runFile: List<File>): Error {
        TODO("Объединение всех файлов " +
                "и производство нового файла классификации")
    }

    fun getName(): String {
        return "Classification"
    }
}