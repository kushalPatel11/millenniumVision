import {
  Controller,
  Post,
  Body,
  HttpException,
  HttpStatus,
} from '@nestjs/common';
import { AuthService } from './auth.service';
import { UserService } from '../user/user.service';

@Controller('auth')
export class AuthController {
  constructor(
    private authService: AuthService,
    private userService: UserService,
  ) {}

  @Post('signup')
  async signup(
    @Body('username') username: string,
    @Body('email') email: string,
    @Body('password') password: string,
    @Body('confirmPassword') confirmPassword: string,
  ) {
    if (password !== confirmPassword) {
      throw new HttpException('Passwords do not match', HttpStatus.BAD_REQUEST);
    }

    const existingUser = await this.userService.findByUsername(username);
    if (existingUser) {
      throw new HttpException('Username already taken', HttpStatus.BAD_REQUEST);
    }

    await this.userService.createUser(username, email, password);
    return { message: 'User created successfully' };
  }

  @Post('login')
  async login(
    @Body('username') username: string,
    @Body('password') password: string,
  ) {
    const user = await this.authService.validateUser(username, password);
    if (!user) {
      throw new HttpException('Invalid credentials', HttpStatus.UNAUTHORIZED);
    }

    return this.authService.login(user);
  }
}
