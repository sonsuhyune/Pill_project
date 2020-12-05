<?php 

    error_reporting(E_ALL); 
    ini_set('display_errors',1); 

    include('dbcon.php');


    $android = strpos($_SERVER['HTTP_USER_AGENT'], "Android");


    if( (($_SERVER['REQUEST_METHOD'] == 'POST') && isset($_POST['submit'])) || $android )
    {

        // 안드로이드 코드의 postParameters 변수에 적어준 이름을 가지고 값을 전달 받습니다.

        $id=$_POST['id']
        $pw=$_POST['pw']
        $name=$_POST['name'];
        $email=$_POST['email'];

        if(empty($id)){
            $errMSG = "아이디를 입력하세요.";
        }
        else if(empty($pw)){
            $errMSG = "비밀번호를 입력하세요.";
        }
        else if(empty($name)){
            $errMSG = "이름을 입력하세요.";
        }
        else if(empty($country)){
            $errMSG = "이메일을 입력하세요.";
        }

        if(!isset($errMSG)) // 모두 입력이 되었다면 
        {
            try{
                // SQL문을 실행하여 데이터를 MySQL 서버의 person 테이블에 저장합니다. 
                $stmt = $con->prepare('INSERT INTO login(id, pw, name, email) VALUES(:id, :pw, :name, :email)');
                $stmt->bindParam(':id', $id);
                $stmt->bindParam(':pw', $pw);
                $stmt->bindParam(':name', $name);
                $stmt->bindParam(':email', $email);

                if($stmt->execute())
                {
                    $successMSG = "새로운 사용자를 추가했습니다.";
                }
                else
                {
                    $errMSG = "사용자 추가 에러";
                }

            } catch(PDOException $e) {
                die("Database error: " . $e->getMessage()); 
            }
        }

    }

?>


<?php 
    if (isset($errMSG)) echo $errMSG;
    if (isset($successMSG)) echo $successMSG;

	$android = strpos($_SERVER['HTTP_USER_AGENT'], "Android");
   
    if( !$android )
    {
?>
    <html>
       <body>

            <form action="<?php $_PHP_SELF ?>" method="POST">
                Name: <input type = "text" name = "name" />
                Country: <input type = "text" name = "country" />
                <input type = "submit" name = "submit" />
            </form>
       
       </body>
    </html>

<?php 
    }
?>