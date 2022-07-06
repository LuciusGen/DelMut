import javax.swing.*;
import javax.swing.filechooser.FileNameExtensionFilter;
import java.awt.*;
import java.io.File;

public class MainForm extends JFrame {
    private JButton runButton;
    private JButton SetFile;
    private JPanel panel;
    private JProgressBar workProgressBar;
    private JCheckBox saveTmp;
    private JTextField nameSnpEffDB;
    private File snpsFile = null;
    private final JFileChooser snipFileChooser = new JFileChooser();
    private final Commands commands = new Commands();
    private final MainFormController controller = new MainFormController(workProgressBar, commands);

    public static void main(String[] args) {
        new MainForm();
    }

    {
        setResizable(false);
        setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        getContentPane().add(panel);
        setMinimumSize(new Dimension(600, 400));
        toFront();
        setVisible(true);
    }

    public MainForm() {
        SetFile.addActionListener(e -> snipFileChooser.showOpenDialog(panel));

        snipFileChooser.addActionListener(e -> snpsFile = snipFileChooser.getSelectedFile());
        runButton.addActionListener(e -> {
            if (snpsFile == null) {
                JOptionPane.showMessageDialog(null, "Set snps file before fun");
            } else {
                Error result = controller.run(snpsFile, !saveTmp.isSelected());
                if (!result.getMessage().isEmpty()){
                    JOptionPane.showMessageDialog(null, result.getMessage());
                } else {
                    JOptionPane.showMessageDialog(null,
                            "Program work end.\n Result is in file: " + result.getValue());
                }
            }
        });

        setFileFilters();
    }

    private void setFileFilters(){
        FileNameExtensionFilter extFilter = new FileNameExtensionFilter ("VCF files (*.vcf)", "*.vcf");
        snipFileChooser.setFileFilter(extFilter);

        snipFileChooser.setAcceptAllFileFilterUsed(false);
    }
}
