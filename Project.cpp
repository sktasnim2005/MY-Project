#include<iostream>
#include<string>
#include<cstdlib>
#include<ctime>
using namespace std;

int main()
{
    cout << "1. Multiplication Table" << endl;
    cout << "2. Calculate Factorial" << endl;
    cout << "3. Prime Number Checker" << endl;
    cout << "4. Reverse a String" << endl;
    cout << "5. Print square using (*) " << endl;
    cout << "6. Guessing Num Game" << endl;
    cout << "7. EXIT" << endl;

    int a;
    cin >> a;
    switch(a) {
        case 1: {
            int n;
            cout << "Enter any integer: ";
            cin >> n;
            for(int i = 1; i <= 10; i++) {
                cout << n << " X " << i << " = " << (n * i) << endl;
            }
            cout << endl;
            break;
        }
        case 2: {
            int n, fact = 1;
            cout << "Enter a number: ";
            cin >> n;
            for(int i = 1; i <= n; i++) {
                fact *= i;
            }
            cout << n << "! = " << fact << endl;
            cout << endl;
            break;
        }
        case 3: {
            int n, count = 0;
            cout << "Enter a number: ";
            cin >> n;
            for(int i = 2; i < n; i++) {
                if(n % i == 0) {
                    count++;
                    break;
                }
            }
            if(count > 0) {
                cout << "Not Prime!" << endl;
            } else {
                cout << "Prime!" << endl;
            }
            cout << endl;
            break;
        }
        case 4: {
            string str;
            cout << "Enter a string: ";
            getline(cin, str);
            int length = str.length();
            for(int i = 0; i < length / 2; i++) {
                swap(str[i], str[length - i - 1]);
            }
            cout << "Reverse = " << str << endl;
            cout << endl;
            break;
        }
        case 5: {
            int n;
            cout << "Enter a number: ";
            cin >> n;
            for(int row = 1; row <= n; row++) {
                for(int col = 1; col <= n; col++) {
                    if(row == 1 || row == n || col == 1 || col == n) {
                        cout << "* ";
                    } else {
                        cout << "  ";
                    }
                }
                cout << endl;
            }
            break;
        }
        case 6: {
            srand(time(0));
            while(1) {
                int guessNum;
                cout << "Enter a number between 1 to 5: ";
                cin >> guessNum;
                int randomNum = 1 + rand() % 5;
                if(guessNum == randomNum) {
                    cout << "You Won!" << endl;
                } else {
                    cout << "You Lost! The correct number was " << randomNum << endl;
                }
                cout << "Do you want to play again? ( yes/no): ";
                string response;
                cin >> response;
                if(response != "yes") {
                    break;
                }
            }
            break;
        }
        case 7: {
            cout << "Exiting..." << endl;
            break;
        }
        default: {
            cout << "Invalid choice!" << endl;
            break;
        }
    }

    return 0;
}
