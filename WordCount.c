#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[]) {
	FILE *fpointer;
	int count = 0;
	if ((fpointer = fopen(argv[2], "r")) == NULL) {
		printf("error!");
		return 0;
	}
	if (argv[1][1] == 'c') {
		char c;
		while ((c = fgetc(fpointer)) != EOF) {
			count++;
		}
		printf("字符数为%d\n", count);
	} else if (argv[1][1] == 'w') {
		char s[100];
		while (fscanf(fpointer, "%s", s) != EOF) {
			count++;
			for (int i = 1; i < strlen(s) - 1; i++) {
				if (s[i] == ',' && s[i - 1] != ',' && s[i + 1] != ',') {
					count++;
				}
			}
		}
		printf("单词数为%d\n", count);
	}
	fclose(fpointer);
	return 0;
}