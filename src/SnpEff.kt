import java.io.File

class SnpEff(
        errorMessage: String,
        command: String
) : AnalysisPipeline {
    private val name: String = "SnpEff"

    override fun run(runFile: File): Error {
        TODO("Нужно ли загружать из дб или нет. Если пользовательская, то геном уже в локалке. Если интернетная, то нужно скачивать" +
                "можно ввести чекбокс на загрузку базы данных"+
        "1) загружаем, не тотлавливая ошибку" +
        "2) пытаемся аннотировать, отлавливаем ошибку или проверяешь, что размер резалт файла > 0")
    }

    override fun getName(): String {
        return name
    }
}
