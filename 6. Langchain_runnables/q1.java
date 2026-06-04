import java.util.Scanner;

public class CaptionAnalyzer {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        String caption = scanner.nextLine().trim();
        scanner.close();

        String[] words = caption.split(" ");

        int wordCount = words.length;
        int charCount = 0;
        for (String word : words) {
            charCount += word.length();
        }

        StringBuilder reversed = new StringBuilder();
        for (int i = words.length - 1; i >= 0; i--) {
            reversed.append(words[i]);
            if (i > 0) reversed.append(" ");
        }

        StringBuilder titleCase = new StringBuilder();
        for (int i = 0; i < words.length; i++) {
            String word = words[i];
            titleCase.append(Character.toUpperCase(word.charAt(0)))
                     .append(word.substring(1).toLowerCase());
            if (i < words.length - 1) titleCase.append(" ");
        }

        System.out.println("Words: " + wordCount);
        System.out.println("Characters: " + charCount);
        System.out.println("Reversed: " + reversed);
        System.out.println("Title Case: " + titleCase);
    }
}