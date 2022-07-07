import telepot
command = 'a'
id = 1693238711
count_handle = 1
count_handle_prev = 0


first_message = "Welcome to home security\n " \
                "- Manage our security\n Tell hi ============> /hi,\n Open the Door   ===> /open" \
                "\n Start smart door ======> /door"
def handle(msg):
    global command, count_handle, chose_command
    content_type, chat_type, chat_id = telepot.glance(msg)
    chat_id = msg['chat']['id']
    command = msg['text']
    print(chat_id)

    count_handle += 1
    if content_type != 'text':
        bot.sendMessage(chat_id, "Wrong command,Type :" + chose_command)
    else:
        command = msg['text']
        print('Got command:', command)
        if command == '/exits':
            print('exit')
        if command == '/hi' or command == '/hi':
            hi()
        if command == '/open':
            openDoor()
        if command == '/close':
            closeDoor()
        if command == '/door':
            doorSystem()


    return command

def hi():
    bot.sendMessage(id, "say hi")
def openDoor():
    bot.sendMessage(id, "Open the Door")
def closeDoor():
    bot.sendMessage(id, "Close the Door")
def doorSystem():
    bot.sendMessage(id, "Smart home run....")

bot = telepot.Bot('5416163618:AAHBBGGW22gxlbFixf06qlqVCIPqnDUKqtU')
bot.sendMessage(id, first_message)
bot.message_loop(handle)
while True:
    if count_handle == 1:
        print('I am Listening....')
        count_handle += 1
        count_handle_prev = 2

    elif count_handle_prev < count_handle:
        print('new syntax')
        print( 'text :', command)


    count_handle_prev = count_handle
