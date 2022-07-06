import javax.swing.*
import java.io.File
import kotlin.collections.List

class MainFormController(
        private val process: JProgressBar,
        private val commands: Commands
) {
    private val pipeline: List<AnalysisPipeline> = listOf(
    )

    private val snpEffPipe = SnpEff(
            errorMessage = "SnpEff error",
            command = commands.snpEffCommand
        )

    private val finalPipeline = clasifierPipeline(
            errorMessage = "Classifier error",
            command = commands.myPythonScript
    )

    private val tmpFiles = mutableListOf<File>()

    private fun runSnpEff(snpFile: File): Error {
        process.value = 1 / (pipeline.size + 2) * 100
        process.string = snpEffPipe.getName()

        return snpEffPipe.run(snpFile)
    }

    private fun runClassifier(files: List<File>): Error {
        process.value = 99
        process.string = finalPipeline.getName()

        return finalPipeline.run(files)
    }

    fun run(snpFile: File, notSaveTmp: Boolean): Error{
        val snpEffRes = runSnpEff(snpFile)

        if (snpEffRes.message.isNotEmpty())
            return snpEffRes

        val file = snpEffRes.value!!

        for (i in pipeline.indices){
            process.value = (i + 1) / (pipeline.size + 2) * 100
            process.string = pipeline[i].getName()

            val tmp = pipeline[i].run(file)
            if (tmp.message.isNotEmpty())
                return tmp
            tmpFiles.add(tmp.value!!)
        }

        if (notSaveTmp){
            tmpFiles.map { it.delete() }
            file.delete()
        }

        return runClassifier(tmpFiles)
    }
}
